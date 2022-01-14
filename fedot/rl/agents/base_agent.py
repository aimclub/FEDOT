import numpy as np
import torch
from torch.autograd.variable import Variable

from fedot.rl.agents.DQN_Agents.DQN import DQN
from fedot.rl.utilities.replay_buffer import ReplayBuffer


class BaseAgent(object):
    def __init__(self, env, input_size, n_actions, batch_size, gamma, epsilon, device='cpu'):
        self.device = device
        self.model = DQN(input_size, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(capacity=1000)
        self.state = env.reset()
        self.losses = []
        self.all_rewards = []
        self.total_reward = 0.0
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, env):
        action = self.model.act(self.state, self.epsilon)

        next_state, reward, done, _ = env.step(action)
        self.replay_buffer.push(self.state, action, reward, next_state, done)

        self.state = next_state
        self.total_reward += reward

        if done:
            self.state = env.reset()
            self.all_rewards.append(self.total_reward)
            self.total_reward = 0

        if len(self.replay_buffer) > self.batch_size:
            loss = self.compute_td_loss(self.batch_size)
            self.losses.append(loss.data[0])

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = Variable(torch.FloatTensor(np.float32(state))).to(self.device)
        action = Variable(torch.LongTensor(action)).to(self.device)
        reward = Variable(torch.FloatTensor(reward)).to(self.device)
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=False).to(self.device)
        done = Variable(torch.FloatTensor(done)).to(self.device)

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma + next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def get_total_reward(self):
        return self.total_reward

    def get_all_rewards(self):
        return self.all_rewards
