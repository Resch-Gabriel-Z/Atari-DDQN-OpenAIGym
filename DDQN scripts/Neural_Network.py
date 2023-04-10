import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self,in_channels,num_actions):
        super(DQN,self).__init__()
        # Define the convolutation layers as Mnih et al. 2015
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels=32,kernel_size=8,stride=4),
            nn.ReLU(),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*1*1,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=num_actions),
        )
    def forward(self,x):
        conv_layers = self.conv(x)
        fc_input = self.flatten(conv_layers)
        action = self.fc(fc_input)
        return action