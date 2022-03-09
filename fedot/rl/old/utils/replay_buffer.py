import random
from collections import namedtuple, deque

import numpy as np
from torch import from_numpy


class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples """

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """

        :param action_size (int): dim of each action
        :param buffer_size (int): max size of buffer
        :param batch_size (int): size of each training batch
        :param seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.exp = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """ Add a new experience to memory """
        temp = self.exp(state, action, reward, next_state, done)
        self.memory.append(temp)

    def sample(self):
        """ "Randomly sample a batch of experiences from memory """
        exp = random.sample(self.memory, k=self.batch_size)

        states = from_numpy(np.vstack([e.state for e in exp if e is not None])).float().to(self.device)
        actions = from_numpy(np.vstack([e.action for e in exp if e is not None])).long().to(self.device)
        rewards = from_numpy(np.vstack([e.reward for e in exp if e is not None])).float().to(self.device)
        next_states = from_numpy(np.vstack([e.next_state for e in exp if e is not None])).float().to(self.device)
        dones = from_numpy(np.vstack([e.done for e in exp if e is not None]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
