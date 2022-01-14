import torch
import torch.nn as nn
import torch.nn.functional as F

class DRQN(nn.Module):
    def __init__(self, input_shape, num_actions, gru_size, body=None, bidir=False, device='cpu'):
        super(DRQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gru_size = gru_size
        self.bidir = bidir
        self.num_directions = 2 if self.bidir else 1
        self.device = device

        self.body = body(input_shape, num_actions)
        self.gru = nn.GRU(
            self.body.feature_size(),
            self.gru_size,
            num_layers=1,
            batch_first=True,
            bidierctional=self.bidir
        )

        # self.fc1 = nn.Linear(self.body.feature_size(), self.gru_size)
        self.fc2 = nn.Linear(self.gru_size, self.num_actions)

    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = x.view((-1,) + self.input_shape)

        feats = self.body(x).view(batch_size, seq_len, -1)
        hidden = self.init_hidden(batch_size) if hx is None else hx

        out, hidden = self.gru(feats, hidden)
        x = self.fc2(out)

        return x, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1 * self.num_directions, batch_size, self.gru_size, device=self.device, dtype=torch.float)