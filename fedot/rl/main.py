import os
from os import makedirs
from os.path import join, exists

import numpy as np
import ptan
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from fedot.core.utils import fedot_project_root, default_fedot_data_dir
from fedot.rl.pipeline_env import PipelineEnv


class PolicyGradientNetwork(nn.Module):
    def __init__(self, n_state, n_action, n_hidden=128):
        super(PolicyGradientNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_action),
        )

    def forward(self, x):
        return self.net(x)


def calc_qvals(rewards, gamma=0.99):
    res = []
    sum_r = 0.0

    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        res.append(sum_r)

    return list(reversed(res))

if __name__ == '__main__':
    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)

    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

    env = PipelineEnv([full_path_train, full_path_test])
    # env = gym.make('CartPole-v1')
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n

    gamma = 0.99

    pnet = PolicyGradientNetwork(in_dim, out_dim)
    print(pnet)

    agent = ptan.agent.PolicyAgent(pnet, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)

    # exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=4, steps_delta=4)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma)

    optimizer = optim.Adam(pnet.parameters(), lr=0.01)

    # Tensorboard
    path_to_tbX = join(default_fedot_data_dir(), 'rl', 'tensorboard')

    if not exists(path_to_tbX):
        makedirs(path_to_tbX)

    tb_writer = SummaryWriter(logdir=path_to_tbX)

    total_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_episode = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = []
    pipeline_length = []
    correct_pipeline = 0
    reward_window = 25

    for step_idx, exp in enumerate(exp_source):
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        if exp.last_state is None:
            batch_qvals.extend(calc_qvals(cur_rewards, gamma))
            cur_rewards.clear()
            batch_episode += 1

        new_rewards = exp_source.pop_total_rewards()

        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            env.render()
            print('%d: reward: %6.2f, mean_100: %6.2f, episodes: %d' % (step_idx, reward, mean_rewards, done_episodes))

            tb_writer.add_scalar('reward', reward, step_idx)
            tb_writer.add_scalar('reward_100', mean_rewards, step_idx)
            tb_writer.add_scalar('episodes', done_episodes, step_idx)

            if mean_rewards > 90:
                print('Solved in %d steps and %d episodes!' % (step_idx, done_episodes))
                break

        if batch_episode < 4:  # EPISODES_TO_TRAIN
            continue

        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        logits_v = pnet(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()

        batch_episode = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    tb_writer.close()
