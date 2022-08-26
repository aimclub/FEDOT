from enum import Enum, auto
from typing import TypeVar, Generic, Dict, Sequence, Optional

import gym
import networkx as nx
import numpy as np
from gym import Env
from gym.spaces import Space, Discrete
import torch as th
import torch.nn as nn
from torch.distributions import Categorical

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode

ObsType = TypeVar('ObsType', bound=Space)
ActionType = TypeVar('ActionType')


class PolicyMutationEnum(Enum):
    AddAction = auto()
    RemoveAction = auto()
    AddState = auto()
    RemoveState = auto()
    MergeStates = auto()


class ModelFactory:
    def get_model(self, obs_shape: th.Size, num_outputs: int) -> nn.Module:
        in_size = int(np.prod(obs_shape))
        model = nn.Sequential(
            nn.Linear(in_size, in_size*2),
            nn.Tanh(),
            nn.Linear(in_size*2, in_size*3),
            nn.Tanh(),
            nn.Linear(in_size*3, in_size),
            nn.Tanh(),
            nn.Linear(in_size, num_outputs),
            nn.Softmax()
        )
        return model


class PolicyNode:
    def __init__(self,
                 obs_shape: th.Size,
                 actions: Sequence[ActionType],
                 model_factory: Optional[ModelFactory] = None):
        self._actions = list(actions)
        model_factory = model_factory or ModelFactory()
        self.model = model_factory.get_model(obs_shape, num_outputs=len(actions))

    def fit(self):
        pass

    def predict(self, observation) -> ActionType:
        # TODO: computation of action probs by the model
        action_probs: Sequence[float] = [1.0 / float(len(self._actions))]  # uniform

        action_distrib = Categorical(action_probs)
        action_index = action_distrib.sample(1)
        action = self._actions[action_index]
        return action

    def transition(self, observation):
        pass


class PolicyGraph(Graph, Generic[ObsType]):

    def __init__(self, action_space: Space, observation_space: ObsType):
        self._state: PolicyNode = None  # current node
        self._graph = nx.DiGraph()
        pass

    # TODO: how will this look like with learning? Like, we fit on the rollout history?
    def fit(self):
        pass

    def transition(self, observation) -> ActionType:
        """On each observation transitions to another state
        and outputs Action that's encoded in the edge between states"""
        # self.action =
        # self._state =
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