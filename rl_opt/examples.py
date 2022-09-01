import gym
from gym import Env

from rl_opt.policy_graph import PolicyGraph, get_initial_graph


def evaluate_policy_graph(env: Env, policy: PolicyGraph):
    n_rollouts = 100
    max_steps = 1000

    for i_rollout in range(n_rollouts):
        observations = []
        actions = []
        rewards = []
        observation, info = env.reset(seed=42)

        for i in range(max_steps):
            next_state = policy.predict(observation)  # User-defined policy function
            action = policy.transition(next_state)

            observation, reward, done, info = env.step(action)
            # env.render()
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)

            if done: break

        # TODO: do some learning

    env.close()


def test_initial_graph():
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1", new_step_api=True)
    policy = get_initial_graph(env.observation_space, env.action_space)
