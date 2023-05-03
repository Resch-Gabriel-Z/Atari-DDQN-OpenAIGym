import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from Agent import Agent
from Agent_Learning import agent_learning
from Atari_Preprocessing import environment_maker
from Hyperparameters import hyperparameters
from Model_saving_loading import load_model_dict, save_model_dict, save_final_model
from Neural_Network import DQN
from Replay_Memory import ReplayMemory

NUMBER_OF_MESSAGES = 1000
NUMBER_OF_CHECKPOINTS = 100

# get device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device, torch.cuda.get_device_name(device))

# Hyperparameters for the agent
agent_hyperparameters = [hyperparameters['initial_eps'], hyperparameters['final_eps'],
                         hyperparameters['eps_decay_steps']]

# Create the environment
env = environment_maker('ALE/Pong-v5')

# Create the meta data
game_name = 'Pong'
path_to_model_save = '/home/gabe/PycharmProjects/Atari-DDQN-OpenAIGym/DDQN_model_dicts'

# Create the Agent and the online Network
agent = Agent(*agent_hyperparameters, 1, env.action_space.n).to_device(device)
online_net = DQN(1, env.action_space.n).to_device(device)

# Initialize the weights of the online net with the policy nets weights
online_net.load_state_dict(agent.policy_net.state_dict())

# Initialize optimizer, loss function
optimizer = optim.SGD(agent.policy_net.parameters(), hyperparameters['learning_rate'])
loss_function = nn.MSELoss()

# Initialize Memory
memory = ReplayMemory(hyperparameters['replay_buffer_size'])

# Initialize starting variables to override
start = 0
total_steps = 0
episode_reward_tracker = []

# Load the model
start, episode_reward_tracker, total_steps, memory = load_model_dict(path=path_to_model_save, name=game_name,
                                                                     policy_net=agent.policy_net, online_net=online_net,
                                                                     optimizer=optimizer, starting_point=start,
                                                                     total_steps=total_steps, memory=memory,
                                                                     episode_reward_tracker=episode_reward_tracker)

# Train the Agent for a number of episodes
for episode in tqdm(range(start, hyperparameters['number_of_episodes'])):
    # First reset the environment and the cumulative reward
    state, _ = env.reset()
    reward_for_episode = 0

    # Then for each step, follow the Pseudocode in the paper
    for step in range(hyperparameters['max_steps_per_episode']):
        state = torch.as_tensor(state).unsqueeze(0).to_device(device)
        action, new_state, reward, done, *others = agent.act(state, env)
        total_steps += 1
        agent.exploration_decay(total_steps=total_steps)

        # Push the memory
        memory.push(state, action, done, new_state, reward)

        # Update the state
        state = new_state

        # add the reward to the cumulative reward
        reward_for_episode += reward

        # Agent Learning
        agent_learning(hyperparameters['batch_size'], hyperparameters['gamma'], memory=memory, agent=agent,
                       online_net=online_net, loss_function=loss_function, optimizer=optimizer, device=device)

        # If a condition arises which makes playing further impossible (such as losing all lives) go to new episode
        if done:
            break

        # Update the online Network
        if total_steps % hyperparameters['target_update_freq'] == 0:
            online_net.load_state_dict(agent.policy_net.state_dict())

    # Update the tracker
    episode_reward_tracker.append(reward_for_episode)

    # Save the Model
    if episode % (hyperparameters['number_of_episodes'] / 10000):
        save_model_dict(path=path_to_model_save, name=game_name, policy_net=agent.policy_net, online_net=online_net,
                        optimizer=optimizer, starting_point=episode, total_steps=total_steps, memory=memory,
                        episode_reward_tracker=episode_reward_tracker)

    # Print out useful information during Training
    if episode % (hyperparameters['number_of_episodes'] / NUMBER_OF_MESSAGES) == 0:
        print(f'\n'
              f'{"~" * 40}\n'
              f'Episode: {episode + 1}\n'
              f'reward for episode: {reward_for_episode}\n'
              f'total steps done: {total_steps}\n'
              f'memory size: {len(memory)}\n'
              f'{"~" * 40}')

# After training, save the models parameters
name_final_model = game_name + '_final'
path_to_final_model = '/home/gabe/PycharmProjects/Atari-DDQN-OpenAIGym/DDQN_model_final_save'
save_final_model(name=name_final_model, path=path_to_final_model, model=agent.policy_net)
df = pd.DataFrame({'cumulative rewards': episode_reward_tracker})
df.to_csv(f'/home/gabe/PycharmProjects/Atari-DDQN-OpenAIGym/media/{game_name}.csv')
