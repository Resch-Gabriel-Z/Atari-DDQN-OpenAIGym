import random

import numpy as np
import torch

from Neural_Network import DQN


class Agent():
    def __init__(self, initial_eps, final_eps, eps_decay_steps, in_channels, num_actions):
        super().__init__()
        # Initialize needed variables to let the Agent 'act' for themselves including the decision when to explore and
        # when to exploit
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.eps_decay_steps = eps_decay_steps
        self.epsilon = initial_eps

        self.in_channels = in_channels
        self.num_actions = num_actions

        # Create the policy net
        self.policy_net = DQN(self.in_channels, self.num_actions)

    # Define act method
    def act(self, state, env):
        # simple e-greedy strategy
        if random.uniform(0, 1) < self.epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(self.policy_net(state))

        # Execute the action in the environment and return important information
        new_state, reward, done, *others = env.step(action)
        return action, new_state, reward, done, others

    # set exploration rate
    def exploration_decay(self, total_steps):
        self.epsilon = np.interp(total_steps, [0, self.eps_decay_steps], [self.initial_eps, self.final_eps])
