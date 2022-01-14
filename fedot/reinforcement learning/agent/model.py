import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None, config=None):
        self.seq_len = config.SEQUENCE_LENGTH

        super(Model, self).__init__(static_policy, env, config)

        self.reset_hx()