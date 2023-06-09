import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    A CNN defined in the paper by Mnih et al. 2015
    """
    def __init__(self, in_channels, num_actions):
        """

        Args:
            in_channels: the dimension of the state we analyze the picture in
            num_actions: the number of action our agent can do in the current state
        """
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions),
        )

    def forward(self, x):
        """
        The normal forward function in a CNN. But first we make sure that the input is a tensor of the correct type.
        Args:
            x: the current state

        Returns:
            the action to peform in that state
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        conv_layers = self.conv(x)
        fc_input = self.flatten(conv_layers)
        fc_input = fc_input.view(-1, 64 * 7 * 7)
        action = self.fc(fc_input)
        return action
