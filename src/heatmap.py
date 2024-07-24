import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def calculate_reward_with_varying_features(traj_data, time_steps, policy_net, discrim_net, env, transit_dict):
    device = torch.device('cpu')
    policy_net.to(device)
    discrim_net.to(device)

    reward_data = []

    # Only use the first step of the first trajectory
    traj = traj_data[0]
    time_step = int(time_steps[0])
    path = traj.split('_')
    
    des = torch.LongTensor([int(path[-1])]).long().to(device)
    state = torch.LongTensor([int(path[0])]).to(device)
    next_state = torch.LongTensor([int(path[1])]).to(device)
    time_step_tensor = torch.LongTensor([time_step]).to(device)
    
    action = transit_dict.get((int(path[0]), int(path[1])), 'N/A')
    action_tensor = torch.LongTensor([action]).to(device) if action != 'N/A' else None

    if action_tensor is not None:
        # Get the input features
        neigh_path_feature, neigh_edge_feature, original_path_feature, edge_feature, next_path_feature, next_edge_feature = discrim_net.get_input_features(state, des, action_tensor, next_state)

        # Get the log probability of the action
        log_prob = policy_net.get_log_prob(state, des, action_tensor, time_step_tensor).squeeze()

        # Iterate through different values for the first and second elements of path_feature
        for path_feature_first_value in np.arange(-1.0, 1.1, 0.1):
            for path_feature_second_value in np.arange(-1.0, 1.1, 0.1):
                # Create a new path_feature tensor with the modified first and second values
                path_feature = original_path_feature.clone()
                path_feature[0] = path_feature_first_value
                path_feature[1] = path_feature_second_value

                # Calculate the reward using the discriminator
                reward = discrim_net.forward_with_actual_features(
                    neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, 
                    action_tensor, log_prob, next_path_feature, next_edge_feature, time_step_tensor
                )

                reward_data.append({
                    'path_feature_first_value': path_feature_first_value,
                    'path_feature_second_value': path_feature_second_value,
                    'reward': reward.item(),
                })

    # Convert reward_data to a pandas DataFrame
    reward_df = pd.DataFrame(reward_data)

    return reward_df

# Load the model
load_model(model_p)

# Read the trajectory data from the CSV file
trajectory_data = []
with open('trajectory_with_timestep.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        trajectory_data.append(row)

# Extract test trajectories and their timesteps
test_traj = [row[0] for row in trajectory_data]
test_time_steps = [row[1] for row in trajectory_data]

# Read the transit data from the CSV file
transit_data = pd.read_csv('../data/transit.csv')

# Create a dictionary to map (link_id, next_link_id) to action
transit_dict = {}
for _, row in transit_data.iterrows():
    transit_dict[(row['link_id'], row['next_link_id'])] = row['action']

# Calculate rewards for varying path_feature first and second values
reward_df = calculate_reward_with_varying_features(test_traj, test_time_steps, policy_net, discrim_net, env, transit_dict)

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=-reward_df['reward'].values.reshape(21, 21),
    x=np.arange(-1.0, 1.1, 0.1),
    y=np.arange(-1.0, 1.1, 0.1),
    colorscale='Inferno'
))

fig.update_layout(
    title="Reward Heatmap for Varying Path Feature Values",
    xaxis_title="First Path Feature Value",
    yaxis_title="Second Path Feature Value",
    width=800,
    height=800,
)

# Show the plot
fig.show()

# Optionally, save to HTML
fig.write_html("./output/reward_heatmap.html")
print("Saved heatmap to ./output/reward_heatmap.html")