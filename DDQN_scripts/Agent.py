from Neural_Network import DQN
import numpy as np
import torch
import random


class Agent():
    def __init__(self,initial_eps,final_eps,eps_decay_steps,in_channels,num_actions):
        super().__init__()

        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.eps_decay_steps = eps_decay_steps
        self.epsilon = initial_eps

        self.in_channels = in_channels
        self.num_actions = num_actions

        # Create the policy net
        self.policy_net = DQN(self.in_channels,self.num_actions)

    # define act method
    def act(self, state):

        if random.uniform(0,1) < self.epsilon:
            action = state.action_space.sample()
        else:
            action = torch.argmax(self.policy_net(state))

        new_state, reward, done, *others = state.step(action)
        return new_state,reward,done,others

    # set exploration rate
    def exploration_decay(self, total_steps):
        self.epsilon = np.interp(total_steps, [0,self.eps_decay_steps], [self.initial_eps,self.final_eps])