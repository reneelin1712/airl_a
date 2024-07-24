import csv
import pandas as pd
import torch
import numpy as np

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

import shap
from sklearn.ensemble import RandomForestRegressor

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
# test_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
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
                        env.pad_idx,speed_data).to(device)
value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],speed_data=speed_data).to(device)
discrim_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
                                    env.state_action, path_feature_pad, edge_feature_pad,
                                    path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                                    path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
                                    env.pad_idx,speed_data).to(device)

# Read the transit data from the CSV file
transit_data = pd.read_csv('../data/transit.csv')

# Create a dictionary to map (link_id, next_link_id) to action
transit_dict = {}
for _, row in transit_data.iterrows():
    transit_dict[(row['link_id'], row['next_link_id'])] = row['action']

# Read the trajectory data from the CSV file
trajectory_data = []
with open('trajectory_with_timestep.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        trajectory_data.append(row)

def evaluate_rewards(traj_data, time_steps, policy_net, discrim_net, env, transit_dict):
    device = torch.device('cpu')  # Use CPU device
    policy_net.to(device)
    discrim_net.to(device)
    
    reward_data = []
    
    for traj, time_step in zip(traj_data, time_steps):
        path = traj.split('_')
        time_step = int(time_step)
        
        des = torch.LongTensor([int(path[-1])]).long().to(device)
        
        step_rewards = []
        for step_idx in range(len(path) - 1):
            state = torch.LongTensor([int(path[step_idx])]).to(device)
            next_state = torch.LongTensor([int(path[step_idx + 1])]).to(device)
            time_step_tensor = torch.LongTensor([time_step]).to(device)
            
            action = transit_dict.get((int(path[step_idx]), int(path[step_idx + 1])), 'N/A')
            action_tensor = torch.LongTensor([action]).to(device) if action != 'N/A' else None
            
            if action_tensor is not None:
                with torch.no_grad():
                    print('time_step_tensor',time_step_tensor)
                    print('action_tensor',action_tensor)
                    log_prob = policy_net.get_log_prob(state, des, action_tensor, time_step_tensor).squeeze()
                    reward = discrim_net.calculate_reward(state, des, action_tensor, log_prob, next_state,time_step_tensor).item()
            else:
                reward = 'N/A'
            
            step_rewards.append(str(reward))
        
        reward_data.append('_'.join(step_rewards))
    
    return reward_data

# Extract test and learner trajectories and their timesteps
test_traj = [row[0] for row in trajectory_data]
test_time_steps = [row[1] for row in trajectory_data]
learner_traj = [row[2] for row in trajectory_data]
learner_time_steps = [row[3] for row in trajectory_data]

# Calculate rewards for test and learner trajectories
test_reward_data = evaluate_rewards(test_traj, test_time_steps, policy_net, discrim_net, env, transit_dict)
learner_reward_data = evaluate_rewards(learner_traj, learner_time_steps, policy_net, discrim_net, env, transit_dict)

# Merge reward data with trajectory data
updated_trajectory_data = []
for (test_traj, test_time_step, learner_traj, learner_time_step), test_reward, learner_reward in zip(trajectory_data, test_reward_data, learner_reward_data):
    test_links = test_traj.split('_')
    learner_links = learner_traj.split('_')
    
    test_actions = []
    learner_actions = []
    
    for i in range(len(test_links) - 1):
        link_id = int(test_links[i])
        next_link_id = int(test_links[i + 1])
        action = transit_dict.get((link_id, next_link_id), 'N/A')
        test_actions.append(str(action))
    
    for i in range(len(learner_links) - 1):
        link_id = int(learner_links[i])
        next_link_id = int(learner_links[i + 1])
        action = transit_dict.get((link_id, next_link_id), 'N/A')
        learner_actions.append(str(action))
    
    test_return = sum(float(r) for r in test_reward.split('_') if r != 'N/A')
    learner_return = sum(float(r) for r in learner_reward.split('_') if r != 'N/A')
    
    updated_trajectory_data.append([
        test_traj, test_time_step, learner_traj, learner_time_step,
        '_'.join(test_actions), '_'.join(learner_actions),
        test_reward, learner_reward,
        test_return, learner_return
    ])

# Save the updated trajectory data with actions, rewards, and returns to a new CSV file
with open('trajectories_with_actions_rewards_returns.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Test Trajectory', 'Test Trajectory Timestep', 'Learner Trajectory', 'Learner Trajectory Timestep',
                         'Test Actions', 'Learner Actions', 'Test Rewards', 'Learner Rewards',
                         'Test Return', 'Learner Return'])
    csv_writer.writerows(updated_trajectory_data)