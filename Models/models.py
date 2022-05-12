import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_FC(nn.Module):
    def __init__(self, input_channels=3, act_num=5, softmax=True):
        super().__init__()
        self.softmax = softmax
        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.cov_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 1024),
        )
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, act_num)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.cov_out(x))
        if self.softmax:
            a = F.softmax(self.actor_linear(x), dim=1)
        else:
            a = self.actor_linear(x)
        return a, self.critic_linear(x)


class Mario(nn.Module):
    def __init__(self, input_channels=3, act_num=5, softmax=True):
        super(Mario, self).__init__()
        self.softmax = softmax
        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, act_num)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.linear(x.view(x.size(0), -1))
        if self.softmax:
            a = F.softmax(self.actor_linear(x), dim=1)
        else:
            a = self.actor_linear(x)
        return a, self.critic_linear(x)