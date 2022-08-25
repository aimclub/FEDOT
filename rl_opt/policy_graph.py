from typing import TypeVar, Generic

import gym
from gym import Env
from gym.spaces import Space, Discrete

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode

ObsType = TypeVar('ObsType', bound=Space)
ActionType = TypeVar('ActionType')


class PolicyNode(GraphNode):
    def __init__(self):
        self.model = None

    pass


class PolicyGraph(Graph, Generic[ObsType]):

    def __init__(self, action_space: Space, observation_space: ObsType):
        self._state = None # current ndoe
        pass

    # TODO: how will this look like with learning? Like, we fit on the rollout history?
    def fit(self):
        pass

    def transition(self, observation) -> ActionType:
        """On each observation transitions to another state
        and outputs Action that's encoded in the edge between states"""
        pass




def evaluate_policy_graph():
    env = gym.make("LunarLander-v2")

    policy = PolicyGraph(env.action_space, env.observation_space)

    observation, info = env.reset(seed=42)
    for _ in range(1000):
        # env.render()
        action = policy.transition(observation)  # User-defined policy function
        observation, reward, done, info = env.step(action)

        if done:
            observation, info = env.reset()
    env.close()