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


ObsType = TypeVar('ObsType', bound=Space)
ActionType = TypeVar('ActionType')
StateKeyType = Hashable

ACTION = 'action'
NODE = 'node'
ACT_PROB = 'probability'


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
            nn.Tanh(),
        )
        return model


class PolicyNode:
    def __init__(self,
                 obs_shape: th.Size,
                 transitions: Sequence[StateKeyType],
                 model_factory: Optional[ModelFactory] = None):
        self.transitions = list(transitions)
        model_factory = model_factory or ModelFactory()
        self.model = model_factory.get_model(obs_shape, num_outputs=len(transitions))

    def fit(self):
        pass

    def predict(self, observation) -> StateKeyType:
        # TODO: computation of action probs by the model
        #  and saving of intermediate info for local learning step
        logits = self.model.forward(observation)
        probs = nn.Softmax()(logits)

        # uniform distribution
        probs: Sequence[float] = [1.0 / float(len(self.transitions))]

        distrib = Categorical(probs)
        index = distrib.sample(1)
        transition = self.transitions[index]
        return transition


class PolicyGraph(Generic[ObsType]):

    def __init__(self,
                 observation_space: ObsType,
                 action_space: Space,
                 state_graph: nx.DiGraph,
                 initial_state: Optional[StateKeyType] = None):

        self._graph = PolicyGraph.to_model_graph(observation_space, action_space, state_graph)
        if initial_state is not None:
            if initial_state not in self._graph:
                raise ValueError(f'Invalid initial state {initial_state} is not found in state graph.')
            self._state = initial_state
        else:
            self._state = list(self._graph)[0]
        self._state_key_counter = len(self._graph)

    @staticmethod
    def to_model_graph(observation_space: ObsType,
                       action_space: Space,
                       state_graph: nx.DiGraph) -> nx.DiGraph:
        model_factory = ModelFactory()
        state_attrs = {}
        for state in state_graph:
            successors = state_graph.successors(state)
            # action_edges = state_graph.out_edges(state, data=True)
            # actions = [attrs[ACTION] for _, dst, attrs in action_edges]
            new_model = PolicyNode(observation_space, successors, model_factory)
            state_attrs[state] = {NODE: new_model}
        model_graph = nx.DiGraph(state_graph)
        nx.set_node_attributes(model_graph, state_attrs)
        return model_graph

    def add_transition(self,
                       source_state: StateKeyType,
                       action: ActionType,
                       dest_state: StateKeyType):
        """Add new transition from source node to the destination node with an action.

        If a transition with such action already exists -- then a duplicate is made.
        If destination state is None -- a new state is created.
        """
        source_node = self._graph[source_state]
        if dest_state not in self._graph:
            pass

    @property
    def _next_state_key(self) -> StateKeyType:
        self._state_key_counter += 1
        state_key = 'S' + str(self._state_key_counter)
        return state_key

    # TODO: how will this look like with learning? Like, we fit on the rollout history?
    def fit(self, observation):
        destination = self.predict(observation)
        self.transition(destination)
        # ...
        # ...
        # ...
        pass

    def predict(self, observation) -> StateKeyType:
        return self._state_node.predict(observation)

    def transition(self, destination: StateKeyType) -> ActionType:
        """On each observation transitions to another state
        and outputs Action that's encoded in the edge between states"""

        successors = self._graph.out_edges(self._state, data=True)
        for _, succ_state, attrs in successors:
            if succ_state == destination:
                self._state = destination
                action = attrs[ACTION]
                return action
        else:
            raise ValueError(f'Destination state <{destination}> is not found'
                             f' in successors of the state <{self._state}>')

    @property
    def _state_node(self) -> PolicyNode:
        return self._graph[self._state][NODE]


def evaluate_policy_graph():
    env = gym.make("LunarLander-v2")

    n_rollouts = 100
    max_steps = 1000
    policy = PolicyGraph(env.observation_space, env.action_space)

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