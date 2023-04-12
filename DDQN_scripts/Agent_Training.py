import gym
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from Agent import Agent
from Agent_Learning import agent_learning
from Atari_Preprocessing import AtariWrapper
from Hyperparameters import hyperparameters
from Model_saving_loading import load_model_dict, save_model_dict
from Neural_Network import DQN
from Replay_Memory import ReplayMemory


# A function to create and preprocess the Environment
def environment_maker(game):
    env_base = gym.make(game)
    env_wrapped = AtariWrapper(env_base)
    return env_wrapped


# Hyperparameters for the agent
agent_hyperparameters = [hyperparameters['initial_eps'], hyperparameters['final_eps'],
                         hyperparameters['eps_decay_steps']]

# Create the environment
env = environment_maker('ALE/Breakout-v5')

# Create the meta data
name = 'Breakout'
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

# Load the model
start, total_steps, memory = load_model_dict(path=path_to_model_save, name=name, policy_state_dict=agent.policy_net,
                                             online_state_dict=online_net, optimizer_state_dict=optimizer, start=start,
                                             total_steps=total_steps, memory_savestate=memory)

# Train the Agent for a number of episodes
for episode in tqdm(range(start, hyperparameters['number_of_episodes'])):
    # First reset the environment
    state, _ = env.reset()

    # Then for each step, follow the Pseudocode in the paper
    for step in range(hyperparameters['max_steps_per_episode']):
        state = torch.tensor(state).unsqueeze(0)
        action, new_state, reward, done, *others = agent.act(state, env)
        total_steps += 1
        agent.exploration_decay(total_steps=total_steps)

        # Push the memory
        memory.push(state, action, done, new_state, reward)

        # Update the state
        state = new_state

        # Agent Learning
        agent_learning(hyperparameters['batch_size'], hyperparameters['gamma'], memory=memory, agent=agent,
                       online_network=online_net, loss_function=loss_function, optimizer=optimizer)

        # If a condition arises which makes playing further impossible (such as losing all lives) go to new episode
        if done:
            break

        # Update the online Network
        if total_steps % hyperparameters['target_update_freq'] == 0:
            online_net.load_state_dict(agent.policy_net.state_dict())

    # Save the Model
    save_model_dict(path=path_to_model_save, name=name, policy_state_dict=agent.policy_net,
                    online_state_dict=online_net, optimizer_state_dict=optimizer, start=episode,
                    total_steps=total_steps, memory_savestate=memory)

    # Print out useful information during Training
    if episode % (hyperparameters['number_of_episodes'] / 10000) == 0:
        print(f'\n'
              f'{"~" * 40}\n'
              f'Episode: {episode + 1}\n'
              f'reward: {reward}\n'
              f'total steps done: {total_steps}\n'
              f'info: {others}\n'
              f'memory size: {len(memory)}\n'
              f'{"~" * 40}')
