import torch.nn as nn


class A2CRnn(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(A2CRnn, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )

        """ Возвращает политику (стратегию) с распределением вероятности по действиям """
        self.policy = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        """ Число, которое приблизительно соответствует ценности состояния """
        self.value = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        x = self.net(state.float())

        return self.policy(x), self.value(x)
