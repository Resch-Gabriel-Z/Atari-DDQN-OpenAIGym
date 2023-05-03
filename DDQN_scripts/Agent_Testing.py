import os

import torch

from Agent import Agent
from Atari_Preprocessing import environment_maker

env = environment_maker('ALE/Pong-v5', render_mode='human')

path = '-'
name = 'Pong' + '_final.pt'

# Load the final model
if os.path.exists(path + '/' + name):
    agent = Agent(0.1, 0.1, 0, 1, env.action_space.n)
    agent.policy_net.load_state_dict(
        torch.load(f'{path}/{name}'))
    agent.policy_net.eval()

    # Test the final model
    state, _ = env.reset()
    while True:
        state = torch.as_tensor(state).unsqueeze(0)
        action, new_state, reward, done, *others = agent.act(state, env)
        state = new_state

        if done:
            state, _ = env.reset()
