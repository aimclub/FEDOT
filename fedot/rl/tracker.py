import sys

import numpy as np


class RewardTracker:
    def __init__(self, writer, stop_reward, episodes_limit):
        self.writer = writer
        self.stop_reward = stop_reward
        self.episodes_limit = episodes_limit

    def __enter__(self):
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, episode, epsilon=None):
        self.total_rewards.append(reward)

        mean_reward = np.mean(self.total_rewards[-100:])

        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon

        print("%d: done %d episode, mean reward %.3f, %s" % (
            episode, len(self.total_rewards), mean_reward, epsilon_str
        ))

        sys.stdout.flush()

        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, episode)

        self.writer.add_scalar("reward_100", mean_reward, episode)
        self.writer.add_scalar("reward", reward, episode)

        if mean_reward > self.stop_reward:
            print("Solved in %d episodes!" % episode)
            return True

        if episode >= self.episodes_limit:
            print("Episodes limit is reached %d : %d!" % episode, self.episodes_limit)
            return True

        return False
