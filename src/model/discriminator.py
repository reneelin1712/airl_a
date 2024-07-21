import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.backends.cudnn.enabled = False
import pandas as pd


class DiscriminatorAIRLCNN(nn.Module):
    def __init__(self, action_num, gamma, policy_mask, action_state, path_feature, link_feature, rs_input_dim,
                 hs_input_dim, pad_idx=None, speed_data=None):
        super(DiscriminatorAIRLCNN, self).__init__()

        # Load speed data
        self.speed_data = speed_data

        self.gamma = gamma
        self.policy_mask = torch.from_numpy(policy_mask).long()
        policy_mask_pad = np.concatenate([policy_mask, np.zeros((policy_mask.shape[0], 1), dtype=np.int32)], 1)
        self.policy_mask_pad = torch.from_numpy(policy_mask_pad).long()
        action_state_pad = np.concatenate([action_state, np.expand_dims(np.arange(action_state.shape[0]), 1)], 1)
        self.action_state_pad = torch.from_numpy(action_state_pad).long()
        self.path_feature = torch.from_numpy(path_feature).float()
        self.link_feature = torch.from_numpy(link_feature).float()
        self.new_index = torch.tensor([7, 0, 1, 6, 8, 2, 5, 4, 3]).long()

        self.pad_idx = pad_idx
        self.action_num = action_num

        # change
        self.conv1 = nn.Conv2d(rs_input_dim, 20, 3, padding=1)  # [batch, 20, 3, 3]  # +1 for weather feature
        self.pool = nn.MaxPool2d(2, 1)  # [batch, 20, 3, 3]
        self.conv2 = nn.Conv2d(20, 30, 2)  # [batch, 30, 1, 1]
        self.fc1 = nn.Linear(30 + self.action_num, 120)  # [batch, 120]
        self.fc2 = nn.Linear(120, 84)  # [batch, 84]
        self.fc3 = nn.Linear(84, 1)  # [batch, 8]

        self.h_fc1 = nn.Linear(hs_input_dim, 120)  # [batch, 120]  # +1 for weather feature
        self.h_fc2 = nn.Linear(120, 84)  # [batch, 84]
        self.h_fc3 = nn.Linear(84, 1)  # [batch, 8]

        # Increase the input dimensions to account for the weather feature
        self.conv1 = nn.Conv2d(rs_input_dim + 1, 20, 3, padding=1)
        self.h_fc1 = nn.Linear(hs_input_dim + 1, 120)

    def to_device(self, device):
        self.policy_mask = self.policy_mask.to(device)
        self.policy_mask_pad = self.policy_mask_pad.to(device)
        self.action_state_pad = self.action_state_pad.to(device)
        self.path_feature = self.path_feature.to(device)
        self.link_feature = self.link_feature.to(device)
        self.new_index = self.new_index.to(device)

    def process_neigh_features(self, state, des, time_step):
        state_neighbor = self.action_state_pad[state]
        neigh_path_feature = self.path_feature[state_neighbor, des.unsqueeze(1).repeat(1, self.action_num + 1),
                             :]
        neigh_edge_feature = self.link_feature[state_neighbor, :]
        neigh_mask_feature = self.policy_mask_pad[state].unsqueeze(-1)  # [batch_size, 9, 1]

        # # Extract weather feature from the first dimension of state_neighbor
        # weather_feature = neigh_path_feature[:, :, 0].unsqueeze(-1).float()
         # Get speed features
        speed_features = []
        for i in range(state.size(0)):
            speed_row = []
            for neighbor in state_neighbor[i]:
                speed = self.speed_data.get((neighbor.item(), time_step[i].item()), 0)  # Default to 0 if not found
                speed_row.append(speed)
            speed_features.append(speed_row)
        speed_feature = torch.tensor(speed_features, dtype=torch.float32, device=state.device).unsqueeze(-1)

        neigh_feature = torch.cat([speed_feature, neigh_path_feature, neigh_edge_feature, neigh_mask_feature], -1)

        # neigh_feature = torch.cat([neigh_path_feature, neigh_edge_feature, neigh_mask_feature],
        #                           -1)
        neigh_feature = neigh_feature[:, self.new_index, :]
        x = neigh_feature.view(state.size(0), 3, 3, -1)
        x = x.permute(0, 3, 1, 2)
        return x

    def process_state_features(self, state, des, time_step):
        path_feature = self.path_feature[state, des, :]  # 实在不行你也可以把第一个dimension拉平然后reshape 一下
        edge_feature = self.link_feature[state, :]

        # # Extract weather feature from the first dimension of path_feature
        # weather_feature = path_feature[:, 0].unsqueeze(-1)
            # Get speed features
        speed_features = []
        for i in range(state.size(0)):
            speed = self.speed_data.get((state[i].item(), time_step[i].item()), 0)  # Default to 0 if not found
            speed_features.append(speed)
        speed_feature = torch.tensor(speed_features, dtype=torch.float32, device=state.device).unsqueeze(-1)

    
        # Concatenate weather_feature, path_feature, and edge_feature
        feature = torch.cat([speed_feature, path_feature, edge_feature], -1)
        # feature = torch.cat([path_feature, edge_feature], -1)  # [batch_size, n_path_feature + n_edge_feature]
        return feature

    def f(self, state, des, act, next_state, time_step):
        """rs"""
        x = self.process_neigh_features(state, des, time_step)
        x = self.pool(F.leaky_relu(self.conv1(x), 0.2))
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.view(-1, 30)  # 到这一步等于是对这个3x3的图提取feature
        x_act = F.one_hot(act, num_classes=self.action_num)
        x = torch.cat([x, x_act], 1)  # [batch_size, 38]
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)  # 我个人的建议是你先把它按照图像处理完
        rs = self.fc3(x)

        """hs"""
        x_state = self.process_state_features(state, des, time_step)
        x_state = F.leaky_relu(self.h_fc1(x_state), 0.2)
        x_state = F.leaky_relu(self.h_fc2(x_state), 0.2)
        x_state = self.h_fc3(x_state)

        """hs_next"""
        next_x_state = self.process_state_features(next_state, des, time_step)
        next_x_state = F.leaky_relu(self.h_fc1(next_x_state), 0.2)
        next_x_state = F.leaky_relu(self.h_fc2(next_x_state), 0.2)
        next_x_state = self.h_fc3(next_x_state)

        return rs + self.gamma * next_x_state - x_state

    def forward(self, states, des, act, log_pis, next_states, time_steps):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, des, act, next_states, time_steps) - log_pis

    def calculate_reward(self, states, des, act, log_pis, next_states, time_steps):
        with torch.no_grad():
            logits = self.forward(states, des, act, log_pis, next_states, time_steps)
            return -F.logsigmoid(-logits)
        

    def get_single_input_features(self, state, des, action, next_state):
        state_neighbor = self.action_state_pad[state]
        neigh_path_feature = self.path_feature[state_neighbor, des.unsqueeze(1).repeat(1, self.action_num + 1), :]
        neigh_edge_feature = self.link_feature[state_neighbor, :]

      
        current_path_feature = self.path_feature[state, des, :]
        current_edge_feature = self.link_feature[state, :]

        next_path_feature = self.path_feature[next_state, des, :]
        print('next_path_feature',next_path_feature)
        next_edge_feature = self.link_feature[next_state, :]

        self.cur_state = state
        return neigh_path_feature, neigh_edge_feature, current_path_feature, current_edge_feature, next_path_feature, next_edge_feature
    
    
    def get_input_features(self, state, des, action, next_state):
        state_neighbor = self.action_state_pad[state]
        neigh_path_feature = self.path_feature[state_neighbor, des.unsqueeze(1).repeat(1, self.action_num + 1), :]
        neigh_edge_feature = self.link_feature[state_neighbor, :]

        # print('state',state)
        # print('des',des)
        # print('neigh_path_feature',neigh_path_feature)
        current_path_feature = neigh_path_feature[:, action, :][0,-1,:]
        # print('current_path_feature',current_path_feature)
        current_edge_feature = neigh_edge_feature[:, action, :][0,-1,:]

        next_path_feature = self.path_feature[next_state, des, :]
        print('next_path_feature',next_path_feature)
        next_edge_feature = self.link_feature[next_state, :]

        self.cur_state = state
        return neigh_path_feature, neigh_edge_feature, current_path_feature, current_edge_feature, next_path_feature, next_edge_feature
    

    def forward_with_actual_features(self, neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, action,log_prob,next_path_feature, next_edge_feature):
        # print('inputs', path_feature)
        # print('inputs', edge_feature)
        # print('shit',shit)
        # Calculate the neigh_mask_feature
        neigh_mask_feature = self.policy_mask_pad[self.cur_state].unsqueeze(-1)  # [batch_size, 9, 1]

        # Process the neighborhood features
        # print('neigh_path_feature',neigh_path_feature)
        # print('neigh_edge_feature',neigh_edge_feature)
        # print('neigh_mask_feature',neigh_mask_feature)
        neigh_feature = torch.cat([neigh_path_feature, neigh_edge_feature, neigh_mask_feature], -1)
        neigh_feature = neigh_feature[:, self.new_index, :]
        x = neigh_feature.view(neigh_path_feature.size(0), 2, 2, -1)
        x = x.permute(0, 3, 1, 2)

        # Pass through the convolutional layers
        x = self.pool(F.leaky_relu(self.conv1(x), 0.2))
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.view(-1, 30)

        # Concatenate with the action feature
        x_act = F.one_hot(action, num_classes=self.action_num)
        x = torch.cat([x, x_act], 1)

        # Pass through the fully connected layers
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        rs = self.fc3(x)

        # Process the current state features
        x_state = torch.cat([path_feature, edge_feature], -1)
        x_state = F.leaky_relu(self.h_fc1(x_state), 0.2)
        x_state = F.leaky_relu(self.h_fc2(x_state), 0.2)
        x_state = self.h_fc3(x_state)

        # Process the next state features
        next_x_state = torch.cat([next_path_feature, next_edge_feature], -1)
        next_x_state = F.leaky_relu(self.h_fc1(next_x_state), 0.2)
        next_x_state = F.leaky_relu(self.h_fc2(next_x_state), 0.2)
        next_x_state = self.h_fc3(next_x_state)

        return rs + self.gamma * next_x_state - x_state - log_prob
    

class DiscriminatorCNN(nn.Module):
    def __init__(self, action_num, policy_mask, action_state, path_feature, link_feature, input_dim, pad_idx=None):
        super(DiscriminatorCNN, self).__init__()
        self.policy_mask = torch.from_numpy(policy_mask).long()
        policy_mask_pad = np.concatenate([policy_mask, np.zeros((policy_mask.shape[0], 1), dtype=np.int32)], 1)
        self.policy_mask_pad = torch.from_numpy(policy_mask_pad).long()
        action_state_pad = np.concatenate([action_state, np.expand_dims(np.arange(action_state.shape[0]), 1)], 1)
        self.action_state_pad = torch.from_numpy(action_state_pad).long()
        self.path_feature = torch.from_numpy(path_feature).float()
        self.link_feature = torch.from_numpy(link_feature).float()
        self.new_index = torch.tensor([7, 0, 1, 6, 8, 2, 5, 4, 3]).long()
        self.pad_idx = pad_idx
        self.action_num = action_num

        self.conv1 = nn.Conv2d(input_dim, 20, 3, padding=1)  # [batch, 20, 3, 3]
        self.pool = nn.MaxPool2d(2, 1)  # [batch, 20, 3, 3]
        self.conv2 = nn.Conv2d(20, 30, 2)  # [batch, 30, 1, 1]
        self.fc1 = nn.Linear(30 + self.action_num, 120)  # [batch, 120]
        self.fc2 = nn.Linear(120, 84)  # [batch, 84]
        self.fc3 = nn.Linear(84, 1)  # [batch, 8]

    def to_device(self, device):
        self.policy_mask = self.policy_mask.to(device)
        self.policy_mask_pad = self.policy_mask_pad.to(device)
        self.action_state_pad = self.action_state_pad.to(device)
        self.path_feature = self.path_feature.to(device)
        self.link_feature = self.link_feature.to(device)
        self.new_index = self.new_index.to(device)

    def process_features(self, state, des):
        state_neighbor = self.action_state_pad[state]
        neigh_path_feature = self.path_feature[state_neighbor, des.unsqueeze(1).repeat(1, self.action_num + 1), :]
        neigh_edge_feature = self.link_feature[state_neighbor, :]
        neigh_mask_feature = self.policy_mask_pad[state].unsqueeze(-1)  # [batch_size, 9, 1]
        neigh_feature = torch.cat([neigh_path_feature, neigh_edge_feature, neigh_mask_feature],
                                  -1)  # [batch_size, 9, n_path_feature + n_edge_feature + 1]
        neigh_feature = neigh_feature[:, self.new_index, :]
        x = neigh_feature.view(state.size(0), 3, 3, -1)
        x = x.permute(0, 3, 1, 2)
        # print('x', x.shape)
        return x

    def forward(self, state, des, act):  # 这是policy
        x = self.process_features(state, des)
        x = self.pool(F.leaky_relu(self.conv1(x), 0.2))
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.view(-1, 30)  # 到这一步等于是对这个3x3的图提取feature

        x_act = F.one_hot(act, num_classes=self.action_num)

        x = torch.cat([x, x_act], 1)  # [batch_size, 38]
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)  # 我个人的建议是你先把它按照图像处理完

        prob = torch.sigmoid(self.fc3(x))
        return prob

    def calculate_reward(self, st, des, act):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -torch.log(self.forward(st, des, act))