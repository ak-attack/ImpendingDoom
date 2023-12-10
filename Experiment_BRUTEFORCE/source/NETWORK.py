import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import skimage.transform

import os
import sys

import itertools as it
import random
from collections import deque
from time import sleep, time
from tqdm import trange

import vizdoom as vzd






#Neural Network Architexture
#===================================================================================================
class DuelQNet(nn.Module):
    """
    This is Duel DQN architecture.
    see https://arxiv.org/abs/1511.06581 for more information.
    """
    def __init__(self, available_actions_count):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(),
        )

        self.state_fc = nn.Sequential(nn.Linear(90, 45), nn.ReLU(), nn.Linear(45, 1))

        self.advantage_fc = nn.Sequential(
            nn.Linear(90, 45), nn.ReLU(), nn.Linear(45, available_actions_count)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        flat_dim = x.shape[1]*x.shape[2]*x.shape[3]
        x = x.view(-1, flat_dim)
        x1 = x[:, :flat_dim//2]  # input for the net to calculate the state value
        x2 = x[:, flat_dim//2:]  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (
            advantage_values - advantage_values.mean(dim=1).reshape(-1, 1)
        )
        return x


