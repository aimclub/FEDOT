import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from fedot.core.utils import fedot_project_root
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

    for episode in range(50000):
        state = env.reset()

        for t in range(200):
            action = pn.act(state)
            state, reward, done, _ = env.step(action)
            pn.rewards.append(reward)
            if done and reward >= 0:
                env.render()
            if done:
                break

        loss = pn.update()
        total_reward = sum(pn.rewards)
        pn.policy_reset()

        print(f'Episode {episode}, loss: {loss}, total_reward: {total_reward}')
