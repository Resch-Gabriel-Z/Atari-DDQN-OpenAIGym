from Neural_Network import DQN
import numpy as np

# act (in traiing meaning it uses an epsilon that will be decayed),
# act (in evaluation meaning that it will always use the best move or has a fixed epsilon),



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
        # TODO (act): Implement e-greedy Strategy
        # TODO (act): transform as tensor
        # TODO (act): manipulate tensor in desired shape/size
        # TODO (act): get max element of tensor (q_value) and return it (probably as .item())
        pass

    # set exploration rate
    def exploration_decay(self, total_steps):
        self.epsilon = np.interp(total_steps, [0,self.eps_decay_steps], [self.initial_eps,self.final_eps])
