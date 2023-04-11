import os

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from Agent import Agent
from Atari_Preprocessing import Atari_wrapper
from Hyperparameters import hyperparameters
from Neural_Network import DQN
from Replay_Memory import ReplayMemory, Memory


# A function to load a model if existing.
def load_model_dict(path, name, **kwargs):
    policy_net_load = kwargs['policy_state_dict']
    online_net_load = kwargs['online_state_dict']
    optimizer_load = kwargs['optimizer_state_dict']
    start_load = kwargs['start']
    total_steps_load = kwargs['total_steps']
    memory_load = kwargs['memory_savestate']

    if os.path.exists(path + '/' + name + '.pth'):
        print('Save File Found!')
        checkpoint = torch.load(path + '/' + name + '.pth')

        policy_net_load.load_state_dict(checkpoint['policy_state_dict'])
        online_net_load.load_state_dict(checkpoint['online_state_dict'])
        start_load = checkpoint['start'] + 1
        optimizer_load.load_state_dict(checkpoint['optimizer_state_dict'])
        total_steps_load = checkpoint['total_steps']
        memory_load = checkpoint['memory_savestate']
    else:
        print('No Save File Found. Beginn new training')

    return start_load, total_steps_load, memory_load


# A function to save a model
def save_model_dict(path, name, **kwargs):
    policy_net_save = kwargs['policy_state_dict']
    online_net_save = kwargs['online_state_dict']
    optimizer_save = kwargs['optimizer_state_dict']
    start_save = kwargs['start']
    total_steps_save = kwargs['total_steps']
    memory_save = kwargs['memory_savestate']

    torch.save({
        'policy_state_dict': policy_net_save.state_dict(),
        'online_state_dict': online_net_save.state_dict(),
        'optimizer_state_dict': optimizer_save.state_dict(),
        'start': start_save,
        'total_steps': total_steps_save,
        'memory_savestate': memory_save,
    }, path + '/' + name + '.pth')


# A function to create and preprocess the Environment
def environment_maker(game):
    env_base = gym.make(game)
    env = Atari_wrapper(env_base)
    return env


# Learning function that updates the policy network
def Agent_learning(batch_size, gamma, **kwargs):
    memory_l = kwargs['memory']
    agent_l = kwargs['agent']
    online_l = kwargs['online_network']
    optimizer_l = kwargs['optimizer']
    loss_func_l = kwargs['loss_function']

    if len(memory_l) < 1000:
        return
    sample_memory = memory_l.sample(batch_size)
    sample_memory_preprocessed = Memory(*zip(*sample_memory))

    # Get each field of the memory sample
    sample_actions = torch.tensor(sample_memory_preprocessed.action, dtype=torch.int64).unsqueeze(-1)
    sample_rewards = torch.tensor(sample_memory_preprocessed.reward, dtype=torch.float32)
    sample_dones = torch.tensor(sample_memory_preprocessed.done, dtype=torch.float32)
    sample_states = torch.stack(sample_memory_preprocessed.state)
    sample_next_states = torch.tensor(np.array(sample_memory_preprocessed.next_state), dtype=torch.float32).unsqueeze(1)

    # compute the target & q values
    max_q_values_online, _ = torch.max(online_l(sample_next_states), dim=1)
    target = sample_rewards + gamma * max_q_values_online * (1 - sample_dones)

    q_values = agent_l.policy_net(sample_states)
    actions_q_values = torch.gather(q_values, dim=1, index=sample_actions)

    # compute loss and backpropagate
    loss = loss_func_l(target.unsqueeze(1), actions_q_values)
    optimizer_l.zero_grad()
    loss.backward()
    optimizer_l.step()


# Hyperparameters
agent_hyperparameters = [hyperparameters['initial_eps'], hyperparameters['final_eps'],
                         hyperparameters['eps_decay_steps']]

# Create the environment
env = environment_maker('ALE/Pong-v5')

# Create the meta data
name = 'Pong'
path_to_model_save = '/home/gabe/PycharmProjects/Atari-DDQN-OpenAIGym/DDQN_model_dicts'

# Create the Agent and the online Network
agent = Agent(*agent_hyperparameters, 1, env.action_space.n)
online_net = DQN(1, env.action_space.n)

# Initialize the weights of the online net with the policy nets weights
online_net.load_state_dict(agent.policy_net.state_dict())

# Initialize optimizer, loss function
optimizer = optim.SGD(agent.policy_net.parameters(), hyperparameters['learning_rate'])
loss_function = nn.MSELoss()

# Initialize Memory
memory = ReplayMemory(hyperparameters['replay_buffer_size'])

if 'start' not in locals():
    start = 0

if 'total_steps' not in locals():
    total_steps = 0


start, total_steps, memory = load_model_dict(path=path_to_model_save, name=name, policy_state_dict=agent.policy_net,
                                             online_state_dict=online_net, optimizer_state_dict=optimizer, start=start,
                                             total_steps=total_steps, memory_savestate=memory)

for episode in tqdm(range(start, hyperparameters['number_of_episodes'])):
    state, _ = env.reset()

    for step in range(hyperparameters['max_steps_per_episode']):
        state = torch.tensor(state).unsqueeze(0)
        action, new_state, reward, done, *others = agent.act(state, env)
        total_steps += 1
        agent.exploration_decay(total_steps=total_steps)

        memory.push(state, action, done, new_state, reward)

        state = new_state

        Agent_learning(hyperparameters['batch_size'], hyperparameters['gamma'], memory=memory, agent=agent,
                       online_network=online_net, loss_function=loss_function, optimizer=optimizer)

        if done:
            break

        if total_steps % hyperparameters['target_update_freq'] == 0:
            online_net.load_state_dict(agent.policy_net.state_dict())

    save_model_dict(path=path_to_model_save, name=name, policy_state_dict=agent.policy_net,
                    online_state_dict=online_net, optimizer_state_dict=optimizer, start=episode,
                    total_steps=total_steps, memory_savestate=memory)

    if episode % (hyperparameters['number_of_episodes'] / 10000) == 0:
        print(f'\n'
              f'{"~" * 40}\n'
              f'Episode: {episode + 1}\n'
              f'reward: {reward}\n'
              f'total steps done: {total_steps}\n'
              f'info: {others}\n'
              f'memory size: {len(memory)}\n'
              f'{"~" * 40}')
