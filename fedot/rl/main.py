import os
from os import makedirs
from os.path import join, exists

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.distributions import Categorical

from fedot.core.utils import fedot_project_root, default_fedot_data_dir
from fedot.rl.pipeline_env import PipelineEnv


class PolicyNetwork:
    def __init__(self, n_state, n_action, n_hidden=64, lr=.001, gamma=0.97):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_action),
            nn.Softmax(),
        )
        self.log_probs = None
        self.rewards = None
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.policy_reset()

    def policy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        return self.model(x)

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.forward(x)
        prob_distribution = Categorical(pdparam)
        action = prob_distribution.sample()
        log_prob = prob_distribution.log_prob(action)
        self.log_probs.append(log_prob)

        return action.item() + 2

    def update(self):
        T = len(self.rewards)
        rets = np.empty(T, dtype=np.float32)
        future_rets = .0

        for t in reversed(range(T)):
            future_ret = self.rewards[t] + self.gamma * future_rets
            rets[t] = future_ret

        rets = torch.tensor(rets)
        log_probs = torch.stack(self.log_probs)
        loss = -log_probs * rets
        loss = torch.sum(loss)
        self.optimizer.step()

        return loss


if __name__ == '__main__':
    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)

    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

    env = PipelineEnv([full_path_train, full_path_test])
    # env = PipelineEnv(path=None)
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n

    pn = PolicyNetwork(in_dim, out_dim)

    # Tensorboard
    path_to_tbX = join(default_fedot_data_dir(), 'fedot', 'rl', 'logs', 'tensorboard')

    if not exists(path_to_tbX):
        makedirs(path_to_tbX)

    tb_writer = SummaryWriter(logdir=path_to_tbX)
    rewards = []
    pipeline_length = []
    correct_pipeline = 0
    reward_window = 25

    for episode in range(25000):
        state = env.reset()
        done = False
        reward = 0

        for t in range(25):
            action = pn.act(state)
            state, reward, done, info = env.step(action)
            pn.rewards.append(reward)
            if done:
                env.render()
                break

        loss = pn.update()
        total_reward = sum(pn.rewards)
        pn.policy_reset()

        rewards.append(reward)

        # Tensorboard
        tb_writer.add_scalar('reward per eps', reward, episode)
        tb_writer.add_scalar('mean reward per eps', np.mean(rewards), episode)
        tb_writer.add_scalar('time steps per eps', info['time_step'], episode)
        tb_writer.add_scalar('metric values per eps', info['metric_value'], episode)

        if episode >= reward_window:  # TODO: changes
            tb_writer.add_scalar('mean window reward', np.mean(rewards[-reward_window:]), episode)

        if info['length'] != -1:
            correct_pipeline += 1
            tb_writer.add_scalar('pipeline length', info['length'], correct_pipeline)
            tb_writer.add_scalar('metric value correct pipelines', info['metric_value'], correct_pipeline)

        print(f'Episode {episode}, loss: {loss}, total_reward: {total_reward}\n')
