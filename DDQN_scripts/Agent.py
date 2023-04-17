import random

import numpy as np
import torch
import torch.nn as nn

from Neural_Network import DQN


class Agent(nn.Module):
    """
    The Agent class it will be responsible for acting in the environment (including returning information and handle
    exploration)
    """

    def __init__(self, initial_eps, final_eps, eps_decay_steps, in_channels, num_actions):
        """
        A simple initialization of important parameters
        Args:
            initial_eps: the probability of exploration we will start at
            final_eps: the minimum exploration we will do
            eps_decay_steps: the number of steps an agent will do in order to get from the initial to the final eps
            in_channels: the number of neurons the state will be analyzed
            num_actions: the number of actions we can take each step
        """
        super().__init__()
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.eps_decay_steps = eps_decay_steps
        self.epsilon = initial_eps

        self.in_channels = in_channels
        self.num_actions = num_actions

        self.policy_net = DQN(self.in_channels, self.num_actions)

    def act(self, state, env):
        """
        A method to decide whether we should explore or exploit (via e-greedy), executing that action, and returning the
        necessary information.
        Args:
            state: the current state we will act on
            env: the environment of the game

        Returns:
            Information for the memory and further calculations
        """
        if random.uniform(0, 1) < self.epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(self.policy_net(state)).item()

        new_state, reward, done, *others = env.step(action)
        return action, new_state, reward, done, others

    def exploration_decay(self, total_steps):
        """
        A simple method that makes sure that we will decay our probability to explore. We use the numpy library for
        interpolation.
        Args:
            total_steps: the steps we have done so far
        """
        self.epsilon = np.interp(total_steps, [0, self.eps_decay_steps], [self.initial_eps, self.final_eps])
