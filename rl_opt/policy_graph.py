from enum import Enum, auto
from typing import TypeVar, Generic, Dict, Sequence, Optional, Hashable, Tuple

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
from fedot.core.log import default_log

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
        action_probs: Sequence[float] = [1.0 / float(len(self._actions))]  # uniform distrib

        action_distrib = Categorical(action_probs)
        action_index = action_distrib.sample(1)
        action = self._actions[action_index]
        return action

    def transition(self, observation):
        pass

    # TODO: add hash for networkx OR come up with some persistent hash-like node keys
    def __hash__(self):
        pass


StateKeyType = Hashable
ACTION = 'action'
ACT_PROB = 'probability'


class PolicyGraph(Graph, Generic[ObsType]):

    def __init__(self, action_space: Space, observation_space: ObsType):
        self._state: StateKeyType = None
        self._graph = nx.DiGraph()
        pass

    # TODO: how will this look like with learning? Like, we fit on the rollout history?
    def fit(self, observation):
        action = self._state_node.predict(observation)
        self.transition(action)

        # ...
        # ...
        # ...

        pass

    def transition(self, action: ActionType):
        """On each observation transitions to another state
        and outputs Action that's encoded in the edge between states"""

        # collect all candidate actions
        successors = self._graph.out_edges(self._state, data=True)
        transition_candidates = []
        transition_probs = []
        for _, succ_state, attrs in successors:
            if attrs[ACTION] == action:
                act_prob = attrs.get(ACT_PROB, None)
                transition_candidates.append(succ_state)
                transition_probs.append(act_prob)

        # if any transition probability is not set, then assume uniform probability
        if not all(transition_probs):
            transition_probs = None
        # choose next state given probabilities
        next_state = np.random.choice(transition_candidates, p=transition_probs)
        self._state = next_state

    @property
    def _state_node(self) -> PolicyNode:
        return self._graph[self._state]




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