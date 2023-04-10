from Neural_Network import DQN

# Define the Agent
# The Agent should be able to:
# act (in traiing meaning it uses an epsilon that will be decayed),
# act (in evaluation meaning that it will always use the best move or has a fixed epsilon),
# decay, meaning that it will decay its epsilon via a decay-strategy


class Agent():
    def __init__(self,initial_eps,final_eps,eps_decay_steps,in_channels,num_actions):
        super().__init__()

        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.eps_decay_steps = eps_decay_steps

        self.in_channels = in_channels
        self.num_actions = num_actions