from utils.evaluation import evaluate_model, evaluate_log_prob, evaluate_train_edit_dist
import time
import torch
from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, \
    minmax_normalization, load_train_sample, load_test_traj
from network_env import RoadWorld
from utils.torch import to_device
import numpy as np
import pandas as pd
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN
import csv

import shap

def load_model(model_path):
    model_dict = torch.load(model_path)
    policy_net.load_state_dict(model_dict['Policy'])
    print("Policy Model loaded Successfully")
    value_net.load_state_dict(model_dict['Value'])
    print("Value Model loaded Successfully")
    discrim_net.load_state_dict(model_dict['Discrim'])
    print("Discrim Model loaded Successfully")

cv = 0  # cross validation process [0, 1, 2, 3, 4]
size = 1000  # size of training data [100, 1000, 10000]
gamma = 0.99  # discount factor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_p = "../trained_models/airl_CV%d_size%d.pt" % (cv, size)
test_p = "../data/cross_validation/test_CV%d.csv" % cv

"""environment"""
edge_p = "../data/edge.txt"
network_p = "../data/transit.npy"
path_feature_p = "../data/feature_od.npy"
train_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
test_p = "../data/cross_validation/test_CV%d.csv" % cv
model_p = "../trained_models/airl_CV%d_size%d.pt" % (cv, size)

"""initialize road environment"""
od_list, od_dist = ini_od_dist(train_p)
env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))
"""load path-level and link-level feature"""
path_feature, path_max, path_min = load_path_feature(path_feature_p)
edge_feature, link_max, link_min = load_link_feature(edge_p)
path_feature = minmax_normalization(path_feature, path_max, path_min)
path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
edge_feature = minmax_normalization(edge_feature, link_max, link_min)
edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
edge_feature_pad[:edge_feature.shape[0], :] = edge_feature

"""define actor and critic"""
edge_data = pd.read_csv('../data/updated_edges.txt')
speed_data = {(row['n_id'], row['time_step']): row['speed'] for _, row in edge_data.iterrows()}

policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
                    path_feature_pad, edge_feature_pad,
                    path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                    env.pad_idx, speed_data).to(device)
value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                    path_feature_pad.shape[-1] + edge_feature_pad.shape[-1], speed_data=speed_data).to(device)
discrim_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
                                env.state_action, path_feature_pad, edge_feature_pad,
                                path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                                path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
                                env.pad_idx, speed_data).to(device)

# Read the transit data from the CSV file
transit_data = pd.read_csv('../data/transit.csv')

# Create a dictionary to map (link_id, next_link_id) to action
transit_dict = {}
for _, row in transit_data.iterrows():
    transit_dict[(row['link_id'], row['next_link_id'])] = row['action']

def evaluate_rewards(traj_data, time_steps, policy_net, discrim_net, env, transit_dict, transit_data):
    device = torch.device('cpu')  # Use CPU device
    policy_net.to(device)
    discrim_net.to(device)
    
    reward_data = []
    all_actions_data = []
    
    for episode_idx, (traj, time_step) in enumerate(zip(traj_data, time_steps)):
        path = traj.split('_')
        time_step = int(time_step)
        
        des = torch.LongTensor([int(path[-1])]).long().to(device)
        
        for step_idx in range(len(path) - 1):
            state = torch.LongTensor([int(path[step_idx])]).to(device)
            next_state = torch.LongTensor([int(path[step_idx + 1])]).to(device)
            time_step_tensor = torch.LongTensor([time_step]).to(device)
            
            action = transit_dict.get((int(path[step_idx]), int(path[step_idx + 1])), 'N/A')
            action_tensor = torch.LongTensor([action]).to(device) if action != 'N/A' else None
            
            if action_tensor is not None:
                with torch.no_grad():
                    neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, next_path_feature, next_edge_feature = discrim_net.get_input_features(state, des, action_tensor, next_state)
                    log_prob = policy_net.get_log_prob(state, des, action_tensor, time_step_tensor).squeeze()
                    reward = discrim_net.forward_with_actual_features(neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, action_tensor, log_prob, next_path_feature, next_edge_feature, time_step_tensor)
            else:
                reward = torch.tensor('N/A')
            
            # Find all possible actions and next states for the current state
            possible_actions = transit_data[(transit_data['link_id'] == int(path[step_idx]))]['action'].tolist()
            possible_next_states = transit_data[(transit_data['link_id'] == int(path[step_idx]))]['next_link_id'].tolist()
            
            # Calculate the reward for each possible action
            possible_rewards = []
            for possible_action, possible_next_state in zip(possible_actions, possible_next_states):
                possible_action_tensor = torch.LongTensor([possible_action]).to(device)
                possible_next_state_tensor = torch.LongTensor([possible_next_state]).to(device)
                
                with torch.no_grad():
                    neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, next_path_feature, next_edge_feature = discrim_net.get_input_features(state, des, possible_action_tensor, possible_next_state_tensor)
                    possible_log_prob = policy_net.get_log_prob(state, des, possible_action_tensor, time_step_tensor).squeeze()
                    possible_reward = discrim_net.forward_with_actual_features(neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, possible_action_tensor, possible_log_prob, next_path_feature, next_edge_feature, time_step_tensor)
                
                possible_rewards.append(possible_reward.item())
            
            reward_data.append({
                'episode': episode_idx,
                'des': des.item(),
                'step': step_idx,
                'state': path[step_idx],
                'action': action,
                'next_state': path[step_idx + 1],
                'reward': reward.item() if reward != 'N/A' else 'N/A',
                'time_step': time_step,
                'neigh_path_feature': neigh_path_feature.detach().cpu().numpy().flatten().tolist() if action_tensor is not None else [],
                'neigh_edge_feature': neigh_edge_feature.detach().cpu().numpy().flatten().tolist() if action_tensor is not None else [],
                'path_feature': path_feature.detach().cpu().numpy().flatten().tolist() if action_tensor is not None else [],
                'edge_feature': edge_feature.detach().cpu().numpy().flatten().tolist() if action_tensor is not None else [],
                'next_path_feature': next_path_feature.detach().cpu().numpy().flatten().tolist() if action_tensor is not None else [],
                'next_edge_feature': next_edge_feature.detach().cpu().numpy().flatten().tolist() if action_tensor is not None else [],
                'log_prob': log_prob.detach().cpu().numpy().item() if action_tensor is not None else 'N/A',
            })
            
            all_actions_data.append([
                episode_idx,
                step_idx,
                int(path[step_idx]),
                action,
                int(path[step_idx + 1]),
                reward.item() if reward != 'N/A' else 'N/A',
                time_step
            ] + [val for pair in zip(possible_actions, possible_next_states, possible_rewards) for val in pair])
    
    # Convert reward_data to a pandas DataFrame
    reward_df = pd.DataFrame(reward_data)
    
    return reward_df, all_actions_data

def save_to_csv(data, filename):
    """Utility function to save data to a CSV file with all features as separate columns."""
    if isinstance(data, pd.DataFrame):
        # Convert list columns to strings
        for col in ['neigh_path_feature', 'neigh_edge_feature', 'path_feature', 'edge_feature', 'next_path_feature', 'next_edge_feature']:
            data[col] = data[col].apply(lambda x: str(x))
        
        data.to_csv(filename, index=False)
        print(f"Saved DataFrame to {filename}")
    else:
        print("Error: data is not a DataFrame")

# Load the model
load_model(model_p)

# Read the trajectory data from the CSV file
trajectory_data = []
with open('trajectory_with_timestep.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        trajectory_data.append(row)

# Extract test and learner trajectories and their timesteps
test_traj = [row[0] for row in trajectory_data]
test_time_steps = [row[1] for row in trajectory_data]
learner_traj = [row[2] for row in trajectory_data]
learner_time_steps = [row[3] for row in trajectory_data]

# Evaluate rewards
reward_df, all_actions_data = evaluate_rewards(test_traj, test_time_steps, policy_net, discrim_net, env, transit_dict, transit_data)

# Saving the reward dataframe to CSV file
reward_csv_path = "./output/reward_data_with_features.csv"
save_to_csv(reward_df, reward_csv_path)

# Save the all actions data to a new CSV file
with open('./output/trajectories_all_actions_rewards.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    header = ['Trajectory ID', 'Step', 'Current State', 'Real Action', 'Next State', 'Real Reward', 'Timestep']
    num_possible_actions = (len(all_actions_data[0]) - 7) // 3
    for i in range(1, num_possible_actions + 1):
        header.extend([f'Possible Action {i}', f'Action {i} Next State', f'Action {i} Reward'])
    
    csv_writer.writerow(header)
    csv_writer.writerows(all_actions_data)