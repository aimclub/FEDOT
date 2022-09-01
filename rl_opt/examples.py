import gym
import torch
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
            next_state = policy.predict(observation)
            action = policy.transition(next_state)

            observation, reward, done, info = env.step(action)
            # env.render()
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)

            if done: break

        # TODO: do some learning

    env.close()



def evaluate_simple(env: Env, policy: PolicyGraph):
    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            obs = torch.as_tensor(obs, dtype=torch.float32)
            # TODO: somehow save choices and pass them instead of actions to learner / loss
            next_state = policy.predict(obs)
            action = policy.transition(next_state)

            # act in the environment
            obs, rew, done, _ = env.step(action)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens


def test_initial_graph():
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1", new_step_api=True)
    policy = get_initial_graph(env.observation_space, env.action_space)
