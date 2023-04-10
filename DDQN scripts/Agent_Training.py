# TODO: create some variables for logging and such

import gym

import torch
import torch.nn as nn
import torch.optim as optim

import os

from Agent import Agent
from Neural_Network import DQN
from Atari_Preprocessing import Atari_wrapper
from Hyperparameters import hyperparameters
from Replay_Memory import ReplayMemory

# A function to load a model if existing
def load_model_dict(path,name,**kwargs):
    # TODO: define the kwargs and load model (currently network state dicts, starting point, optimizer_dict, total steps)
    policy_net_load = kwargs['policy_state_dict']
    online_net_load = kwargs['online_state_dict']
    start_load = kwargs['start']
    optimizer_load = kwargs['optimizer_state_dict']
    total_steps_load = kwargs['total_steps']

    if os.path.exists(path+'/'+name):
        checkpoint = torch.load(path+'/'+name)
    pass

# A function to save a model
def save_model_dict(path,name, **kwargs):
    # TODO: as above
    policy_net_save = kwargs['policy_state_dict']
    online_net_save = kwargs['online_state_dict']
    start_save = kwargs['start']
    optimizer_save = kwargs['optimizer_state_dict']
    total_steps_save = kwargs['total_steps']
    pass

# A function to create and preprocess the Environment
def environment_maker(game):
    env_base = gym.make(game)
    env = Atari_wrapper(env_base)
    return env

# Learning function that updates the policy network
def Agent_learning(batch_size,gamma,**kwargs):
    memory_l = kwargs['memory']
    agent_l = kwargs['agent']
    online_l = kwargs['online_network']
    optimizer_l = kwargs['optimizer']
    loss_func_l = kwargs['loss_function']

    if len(memory_l) < 1000:
        return

    # TODO: get the information from the memory in tensors of shape (batch_size,in_channels)
    # TODO: compute targets with the online network and the formula given in paper
    # TODO: compute loss and optimize policy network
    pass

# Hyperparameters
agent_hyperparameters = [hyperparameters['initial_eps'],hyperparameters['final_eps'],hyperparameters['eps_decay_steps']]


# Create the environment
env = environment_maker("ALE/Breakout-v5")

# Create the Agent and the online Network
agent = Agent(*agent_hyperparameters, env.observation_space.shape, env.action_space.shape)
online_net = DQN(env.observation_space.shape,env.action_space.shape)

# Initialize the weights of the online net with the policy nets weights
online_net.load_state_dict(agent.policy_net.load_state_dict())

# Initialize optimizer, loss function
optimizer = optim.SGD(agent.policy_net.parameters(), hyperparameters['learning_rate'])
loss_function = nn.MSELoss()

# Initialize Memory
memory = ReplayMemory(hyperparameters['replay_buffer_size'])
