import random

import numpy as np
import torch
import torch.nn.functional as F

from fedot.rl.agents.DQN import DQN
from fedot.rl.utils.replay_buffer import ReplayBuffer


class BaseAgent(object):
    def __init__(self, state_size, action_size, dqn_type='DQN', replay_memory_size=1e5, batch_size=64, gamma=0.99,
                 learning_rate=1e-3, target_tau=2e-3, update_rate=4, seed=0, device='cpu'):
        """
        DQN's family Parameters

        :param state_size (int): dim of each state
        :param action_size (int): dim of each action
        :param dqn_type (string): 'DQN' for vanilla QNetwork
        :param replay_memory_size (int): size of replay memory of buffer (ussually 5e4 to 5e6)
        :param batch_size (int): size of the memory batch used for model updates
        :param gamma (float): discounted value of future rewards (usually from .90 to .99)
        :param learning_rate (float): the rate of model learning
        :param target_tau (float): interpolation parameter for weights soft update
        :param update_rate (int): network weight will be update after chosen rate
        :param seed (int): random seed for initializing training point
        """
        self.state_size = state_size
        self.action_size = action_size
        self.dqn_type = dqn_type
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.seed = random.seed(seed)
        self.device = device

        """
        DQN Agent Q-Network
        # For DQN training, two nerual network models are employed;
        # (a) A network that is updated every (step % update_rate == 0)
        # (b) A target network, with weights updated to equal the network at a slower (target_tau) rate.
        # The slower modulation of the target network weights operates to stabilize learning.
        """

        self.network = DQN(state_size, action_size, seed).to(device) if self.dqn_type == 'DQN' else None
        self.target_network = DQN(state_size, action_size, seed).to(device) if self.dqn_type == 'DQN' else None
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.batch_size, self.batch_size, seed, device)

        # Time step
        self.time_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.time_step = (self.time_step + 1) % self.update_rate

        if self.time_step == 0:
            #  If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                exp = self.memory.sample()
                self.learn(exp, self.gamma)

    def act(self, state, eps=0.0):
        """
        Returns actions for given state as per current policy

        :param state: current state
        :param eps: rate for epsilon-greedy action selection
        :return: action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.network.eval()

        with torch.no_grad():
            action_values = self.network(state)

        self.network.train()

        # Epsilon-greedy action selection
        # TODO: add another method for action selection, e.g. Monte-Carlo method
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, exp, gamma):

        states, actions, rewards, next_states, dones = exp

        # Get Q values from current observations (s, a) using model network
        q_sa = self.network(states).gather(1, actions)

        # Regular (Vanilla) DQN
        #  Get max Q values for (s', a') from target model
        q_sa_prime_target_values = self.target_network(next_states).detach()
        q_sa_prime_targets = q_sa_prime_target_values.max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        q_sa_targets = rewards + (gamma * q_sa_prime_targets * (1 - dones))

        # Compute loss
        loss = F.mse_loss(q_sa, q_sa_targets)

        # Minimize loss & network backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.network, self.target_network, self.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """
        Soft update model parameters: W_target = τ * W_local + (1 - τ) * W_target
        :param local_model: weights will be copied from
        :param target_model: weights will be copied to
        :param tau: interpolation parameter
        :return:
        """
        for target_params, local_params in zip(target_model.parameters(), local_model.parameters()):
            target_params.data.copy_(tau * local_params.data + (1.0 - tau) * target_params.data)
