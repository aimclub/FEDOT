from os import walk, makedirs
from os.path import join, exists

import numpy as np
import ptan
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from ptan.agent import default_states_preprocessor
from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm

from fedot.core.utils import fedot_project_root
from fedot.rl.network import A2CRnn
from fedot.rl.pipeline_env import PipelineGenerationEnvironment, EnvironmentDataLoader
from fedot.rl.tracker import RewardTracker

GAMMA = 0.99

LEARNING_RATE = 0.002
ENTROPY_BETA = 0.05
BATCH_SIZE = 25
NUM_ENV = 4
REWARD_STEPS = 2
CLIP_GRAD = 0.1


def declare_environment(name, loader, logdir):
    return PipelineGenerationEnvironment(
        dataset_name=name,
        dataset_loader=loader,
        logdir=logdir,
    )


def unpack_batch(batch, net, device):
    """ Convert batch into training tensors

        :param batch:
        :param net:
        :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []

    # Проходим по обучающему набору переходов и копируем их поля в списки
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)  # Дисконтированная награда
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    # Заводим переменные для вычисления на Torch
    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        # Подготавливаем переменную с последним состоянием в цепочке переходов
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        # Запрашиваем аппроксимацию V(s)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        # Добавляем значение к дисконтированному вознаграждению
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v


if __name__ == '__main__':
    experiment_name = str(input())

    path_to_logdir = join(fedot_project_root(), 'fedot/rl/history/' + experiment_name)
    path_to_train = join(fedot_project_root(), 'fedot/rl/data/train/')
    path_to_valid = join(fedot_project_root(), 'fedot/rl/data/valid/')

    datasets = [file_name for (_, _, file_name) in walk(path_to_train)][0]

    env_dl = EnvironmentDataLoader(path_to_train, path_to_valid)
    envs = []

    for dataset_name in datasets:
        for _ in range(NUM_ENV):
            envs.append(declare_environment(dataset_name, env_dl, path_to_logdir))

    in_dim = envs[0].observation_space.shape[0]
    out_dim = envs[0].action_space.n

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = A2CRnn(in_dim, out_dim).to(device)
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    # Tensorboard
    path_to_tbX = join(path_to_logdir, 'tensorboard')
    if not exists(path_to_tbX):
        makedirs(path_to_tbX)

    # Save model
    path_to_checkpoint = join(path_to_logdir, 'checkpoint')
    if not exists(path_to_checkpoint):
        makedirs(path_to_checkpoint)

    tb_writer = SummaryWriter(log_dir=path_to_tbX)
    batch = []

    with RewardTracker(tb_writer, stop_reward=0.9, episodes_limit=20000) as tracker:
        with ptan.common.utils.TBMeanTracker(tb_writer, batch_size=1) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                new_rewards = exp_source.pop_total_rewards()

                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                if (step_idx % 100) == 0 and step_idx != 0:
                    path_to_save = join(path_to_checkpoint, f'agent_{step_idx}')
                    torch.save(net.state_dict(), path_to_save)

                    with torch.no_grad():
                        for val_id, val_name in enumerate(tqdm(datasets)):
                            val_name = val_name[:-4]
                            val_total_rewards = []
                            val_correct_rewards = []
                            val_correct_pipelines = 0

                            for episode in range(25):
                                state = envs[val_id * NUM_ENV].reset()
                                episode_reward = 0.0
                                done = False

                                while not done:
                                    state = default_states_preprocessor(state).to(device)
                                    logits_v, _ = net(state)

                                    prob_v = F.softmax(logits_v)
                                    prob = prob_v.data.cpu().numpy()
                                    action = np.random.choice(len(prob), p=prob)
                                    state, reward, done, info = envs[val_id * NUM_ENV].step(action, mode='valid')
                                    episode_reward += reward

                                val_total_rewards.append(episode_reward)

                                if info['is_correct']:
                                    val_correct_pipelines += 1
                                    val_correct_rewards.append(episode_reward)
                                    tb_writer.add_scalar(val_name + '_metric', info['metric_value'],
                                                         val_correct_pipelines)
                                    tb_writer.add_scalar(val_name + '_length', info['length'], val_correct_pipelines)

                            tb_writer.add_scalar(val_name + '_reward', np.mean(val_total_rewards), step_idx)
                            tb_writer.add_scalar(val_name + '_pos_reward', np.mean(val_correct_rewards), step_idx)
                            tb_writer.add_scalar(val_name + '_correct_pipelines', val_correct_pipelines, step_idx)

                            envs[val_id * NUM_ENV].reset()

                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
                batch.clear()

                optimizer.zero_grad()

                logits_v, value_v = net(states_v)
                # Рассчитываем MSE между значением, возвращенным сетью, и аппроксимацией, выполненной
                # с помощью уравнения Беллмана, развернутного на REWARD_STEPS вперед
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                # Рассчитываем потери, связанные со стратегией, чтобы выполнить градиенты по стратегиям.
                # Два первых шага заключаются в вычислении логарифма вероятности действий по формуле:
                # A(s, a) = Q(s, a) - V(s)
                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.squeeze(-1).detach()
                # Вычисляем логарифм вероятностей для выбранных действий
                # и масштабируем их с помощью adv_v.
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                # Значение потерь при PG будет равно взятому с обратным знаком среднему для данных
                # масштабированных логарифмов вероятностей
                loss_policy_v = -log_prob_actions_v.mean()

                # Рассчитываем потери на энтропию
                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                # Просчитываем градиенты
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
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
