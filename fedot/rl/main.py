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
        self.path_to_save = join(default_fedot_data_dir(), 'rl', 'checkpoint')
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
    learning_rate = 0.01
    entropy_beta = 0.01
    batch_size = 8
    reward_steps = 10

    pnet = PolicyGradientNetwork(in_dim, out_dim)
    print(pnet)

    agent = ptan.agent.PolicyAgent(pnet, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma, steps_count=reward_steps)

    optimizer = optim.Adam(pnet.parameters(), lr=learning_rate)

    # Tensorboard
    path_to_tbX = join(default_fedot_data_dir(), 'rl', 'tensorboard')
    if not exists(path_to_tbX):
        makedirs(path_to_tbX)

    # Save model
    path_to_checkpoint = join(default_fedot_data_dir(), 'rl', 'checkpoint')
    if not exists(path_to_checkpoint):
        makedirs(path_to_checkpoint)

    tb_writer = SummaryWriter(logdir=path_to_tbX)

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0

    best_mean_rewards = 0
    # last_reward = 0
    # same_reward = 0

    batch_states, batch_actions, batch_scales = [], [], []

    for step_idx, exp in enumerate(exp_source):
        reward_sum += exp.reward
        baseline = reward_sum / (step_idx + 1)
        tb_writer.add_scalar('baseline', baseline, step_idx)

        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)

        # handle new rewards
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

            # if mean_rewards > best_mean_rewards:
            #     best_mean_rewards = mean_rewards
            #     torch.save(pnet.state_dict(), path_to_checkpoint)

            # if numpy.isclose(reward, last_reward, rtol=0.001):
            #     same_reward += 1
            # else:
            #     same_reward = 0
            #
            # if same_reward == 10:
            #     print('Early stopped in %d steps and %d episodes.' % (step_idx, done_episodes))
            #     break

            if mean_rewards > 90:
                print('Solved in %d steps and %d episodes!' % (step_idx, done_episodes))
                break

        if len(batch_states) < batch_size:
            continue

        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales)

        optimizer.zero_grad()

        logits_v = pnet(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(batch_size), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -entropy_beta * entropy_v
        loss_v = loss_policy_v + entropy_loss_v

        loss_v.backward()
        optimizer.step()

        # calc KL-div
        new_logits_v = pnet(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        tb_writer.add_scalar('kl', kl_div_v.item(), step_idx)

        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0

        for p in pnet.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1

        tb_writer.add_scalar("entropy", entropy_v.item(), step_idx)
        tb_writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
        tb_writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
        tb_writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
        tb_writer.add_scalar("loss_total", loss_v.item(), step_idx)
        tb_writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
        tb_writer.add_scalar("grad_max", grad_max, step_idx)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

        # last_reward = reward

    tb_writer.close()
