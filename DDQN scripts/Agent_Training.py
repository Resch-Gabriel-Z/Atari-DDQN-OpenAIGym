# TODO: define load function that searches for a file and loads the data in the premade variables
# TODO: define those variables beforehand
# TODO: define save functions that saves the file
# TODO: define learning function
# TODO: define an update target function

# TODO: create an Agent and an online network (and initialize weights)
# TODO: create the Memory
# TODO: create optimizer and loss function
# TODO: create some variables for logging and such

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from Agent import Agent
from Neural_Network import DQN
from Atari_Preprocessing import Atari_wrapper
from Hyperparameters import hyperparameters
from Replay_Memory import ReplayMemory

# A function to load a model if existing
def load_model_dict(path,name):
    pass

# A function to save a model
def save_model_dict(path,name):
    pass

# A function to create and preprocess the Environment
def environment_maker(game):
    env_base = gym.make(game)
    env = Atari_wrapper(env_base)
    return env

# Learning function that updates the policy network
def Agent_learning(batch_size):
    pass

def update_online_network():
    pass

# Hyperparameters
agent_hyperparameters = [hyperparameters['initial_eps'],hyperparameters['final_eps'],hyperparameters['eps_decay_steps']]


# Create the environment
env = environment_maker("ALE/Breakout-v5")

# Create the Agent and the online Network
policy_net = Agent(*agent_hyperparameters,env.observation_space.shape,env.action_space.shape)
online_net = DQN(env.observation_space.shape,env.action_space.shape)

# Initialize the weights of the online net with the policy nets weights
online_net.load_state_dict(policy_net.policy_net.load_state_dict())

# Initialize optimizer, loss function
optimizer = optim.SGD(policy_net.policy_net.parameters(),hyperparameters['learning_rate'])
loss_function = nn.MSELoss()

# Initialize Memory
memory = ReplayMemory(hyperparameters['replay_buffer_size'])
