from itertools import product

import numpy as np
import torch

from fedot.rl.agents.base_agent import BaseAgent
from fedot.rl.environments.graph_env import GraphEnv

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_episodes = 5
    num_nodes = 4
    input_nodes = 1
    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.99
    scores = []
    scores_avg_window = 100

    action_list = list(product(range(num_nodes), range(num_nodes)))

    env = GraphEnv(network_size=num_nodes, input_nodes=input_nodes)

    agent = BaseAgent(
        state_size=env.get_observation_size(),
        action_size=env.get_action_size(),
        device=device
    )

    for episode in range(1, num_episodes + 1):
        state = env.reset()

        episode_score = 0

        env.render_truth()

        while True:
            idx_action = agent.act(state, eps)
            action = action_list[idx_action]

            next_state, reward, done, info = env.step(action)

            agent.step(state, idx_action, reward, next_state, done)

            state = next_state

            episode_score += reward

            env.render()

            if done:
                break

        scores.append(episode_score)
        avg_score = np.mean(scores[episode - min(episode, scores_avg_window):episode + 1])

        eps = max(eps_min, eps_decay * eps)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, avg_score), end="")
