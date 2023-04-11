# TODO: create some variables for logging and such
import random

import gym

import torch
import torch.nn as nn
import torch.optim as optim

import os


from Agent import Agent
from Neural_Network import DQN
from Atari_Preprocessing import Atari_wrapper
from Hyperparameters import hyperparameters
from Replay_Memory import ReplayMemory,Memory

# A function to load a model if existing.
def load_model_dict(path,name,**kwargs):
    policy_net_load = kwargs['policy_state_dict']
    online_net_load = kwargs['online_state_dict']
    start_load = kwargs['start']
    optimizer_load = kwargs['optimizer_state_dict']
    total_steps_load = kwargs['total_steps']

    if os.path.exists(path+'/'+name):
        checkpoint = torch.load(path+'/'+name)

        policy_net_load.load_state_dict(checkpoint['policy_state_dict'])
        online_net_load.load_state_dict(checkpoint['online_state_dict'])
        start_load = checkpoint['start'] + 1
        optimizer_load.load_state_dict(checkpoint['optimizer_state_dict'])
        total_steps_load = checkpoint['total_steps']


# A function to save a model
def save_model_dict(path,name, **kwargs):
    policy_net_save = kwargs['policy_state_dict']
    online_net_save = kwargs['online_state_dict']
    start_save = kwargs['start']
    optimizer_save = kwargs['optimizer_state_dict']
    total_steps_save = kwargs['total_steps']

    torch.save({
        'policy_state_dict' : policy_net_save.state_dict(),
        'online_state_dict' : online_net_save.state_dict(),
        'optimizer_state_dict': optimizer_save.state_dict(),
        'start': start_save,
        'total_steps': total_steps_save
    },path+'/'+name)


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

    sample_memory = random.sample(memory_l)
    sample_memory_preprocessed = Memory(*zip(*sample_memory))

    # Get each field of the memory sample
    sample_actions = torch.tensor(sample_memory_preprocessed.action)
    sample_rewards = torch.tensor(sample_memory_preprocessed.reward)
    sample_dones = torch.tensor(sample_memory_preprocessed.done)
    sample_states = torch.tensor(sample_memory_preprocessed.state)
    sample_next_states = torch.tensor(sample_memory_preprocessed.next_state)

    # compute the target & q values
    max_q_values_online, _ = torch.max(online_l(sample_next_states),dim=1)
    target = sample_rewards + hyperparameters['gamma'] * max_q_values_online * (1-sample_dones)

    q_values = agent_l.policy_net(sample_states)
    actions_q_values = torch.gather(q_values,1,sample_actions)

    # compute loss and backpropagate
    loss = loss_func_l(target,actions_q_values)
    optimizer_l.zero_grad()
    loss.backward()
    optimizer_l.step()


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

