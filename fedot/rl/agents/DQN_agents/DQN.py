import random

import torch
import torch.autograd as autograd
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, device='cpu'):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.device = device

        self.layers = nn.Sequential(
            nn.Linear(self.input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = autograd.Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True).to(self.device)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.n_actions)

        return action
