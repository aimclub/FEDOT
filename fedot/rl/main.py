import os
import sys
import time
from os import makedirs
from os.path import join, exists

import numpy as np
import ptan
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from fedot.core.utils import fedot_project_root, default_fedot_data_dir
from fedot.rl.pipeline_env import PipelineEnv

GAMMA = 0.99

LEARNING_RATE = 0.003
ENTROPY_BETA = 0.3
BATCH_SIZE = 50
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1


class A2CRnn(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(A2CRnn, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state):
        x = self.net(state.float())

        return self.policy(x), self.value(x)


class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward
        self.ts_frame = None
        self.ts = None

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


def unpack_batch(batch, net, device):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v


if __name__ == '__main__':
    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)

    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

    env = PipelineEnv([full_path_train, full_path_test])
    # env = gym.make('CartPole-v1')
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # pnet = PolicyGradientNetwork(in_dim, out_dim)
    # print(pnet)

    pnet = A2CRnn(in_dim, out_dim).to(device)
    print(pnet)

    agent = ptan.agent.PolicyAgent(lambda x: pnet(x)[0], apply_softmax=True, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(pnet.parameters(), lr=LEARNING_RATE, eps=1e-3)

    # Tensorboard
    path_to_tbX = join(default_fedot_data_dir(), 'rl', 'tensorboard')
    if not exists(path_to_tbX):
        makedirs(path_to_tbX)

    # Save model
    # path_to_checkpoint = join(default_fedot_data_dir(), 'rl', 'checkpoint')
    # if not exists(path_to_checkpoint):
    #     makedirs(path_to_checkpoint)

    tb_writer = SummaryWriter(log_dir=path_to_tbX)

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0

    best_mean_rewards = 0
    # last_reward = 0
    # same_reward = 0

    batch = []

    with RewardTracker(tb_writer, stop_reward=18) as tracker:
        with ptan.common.utils.TBMeanTracker(tb_writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()

                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_t, vals_ref_v = unpack_batch(batch, pnet, device=device)
                batch.clear()

                optimizer.zero_grad()

                logits_v, value_v = pnet(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.squeeze(-1).detach()
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                # Calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in pnet.parameters()
                                        if p.grad is not None])

                # Apply entropy and value gradients
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn.utils.clip_grad_norm_(pnet.parameters(), CLIP_GRAD)
                optimizer.step()

                loss_v += loss_policy_v

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var", np.var(grads), step_idx)
