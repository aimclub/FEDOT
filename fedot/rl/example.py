from itertools import product

import numpy as np
import torch

from fedot.rl.agents.base_agent import BaseAgent
from fedot.rl.environments.graph_env import GraphEnv

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_episodes = 2000
    num_nodes = 5
    input_nodes = 2
    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.99
    scores = []
    times = []
    avg_scores_list = []
    avg_times_list = []
    scores_avg_window = 50

    action_list = list(product(range(num_nodes), range(num_nodes))) * 2

    env = GraphEnv(network_size=num_nodes, input_nodes=input_nodes)

    agent = BaseAgent(
        state_size=env.get_observation_size(),
        action_size=env.get_action_size(),
        device=device
    )

    for episode in range(1, num_episodes + 1):
        state = env.reset()

        episode_score = 0
        done = 0

        while True:
            idx_action = agent.act(state, eps)

            if idx_action < len(action_list) / 2:
                i, j = action_list[idx_action]
                action = tuple([i, j, 0])
            else:
                i, j = action_list[idx_action]
                action = tuple([i, j, 1])
            try:
                next_state, reward, done, info = env.step(action)

                agent.step(state, idx_action, reward, next_state, done)

                state = next_state

                episode_score += reward

                if episode == num_episodes:
                    env.render()

            except:
                episode_score -= 0.25

            if done:
                if episode == num_episodes:
                    env.render_truth()
                    env.render()
                break

        scores.append(episode_score)
        times.append(info['time_step'])
        avg_score = np.mean(scores[episode - min(episode, scores_avg_window):episode + 1])
        avg_time = np.mean(times[episode - min(episode, scores_avg_window):episode + 1])

        eps = max(eps_min, eps_decay * eps)

        print('\rEpisode {}\tAverage Score: {:.2f}, {}'.format(episode, avg_score, info['time_step']), end="")

    print('\n')
    print(scores)
    print(times)
