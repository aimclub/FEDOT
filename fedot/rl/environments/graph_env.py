from itertools import product

import gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gym import spaces
from gym.utils import seeding


class GraphEnv(gym.Env):
    """

    """
    metadata = {'render.modes': ['human', 'graph', 'interactive']}

    def __init__(self, network_size=10, input_nodes=3):
        self.network_size = network_size
        self.input_nodes = input_nodes
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(network_size))

        self.action_space = spaces.Tuple((spaces.Discrete(network_size), spaces.Discrete(network_size)))
        self.observation_space = spaces.MultiDiscrete(np.full((self.network_size, self.network_size), 2))
        self.time_step = 0
        self.observation = nx.to_numpy_matrix(self.graph).astype(int)
        self.seed_value = self.seed()
        self.true_graph = self.create_true_graph()
        self.reset()

    def create_true_graph(self):
        final_workflow = nx.DiGraph()
        final_workflow.add_nodes_from(range(self.network_size))
        i = 0
        while nx.ancestors(final_workflow, self.network_size - 1) != set(range(self.network_size - 1)):
            i += 1
            if i > 10000:
                raise RuntimeError('generating graph took too long')
            valid_source_nodes = [index for index, in_degree in
                                  final_workflow.in_degree() if
                                  ((in_degree > 0 or index < self.input_nodes) and index < (self.network_size - 1))]
            valid_to_nodes = [index for index in range(self.input_nodes, self.network_size)]
            new_edge = [(self.np_random.choice(valid_source_nodes), self.np_random.choice(valid_to_nodes))]
            final_workflow.add_edges_from(new_edge)
            if not nx.algorithms.dag.is_directed_acyclic_graph(final_workflow):
                final_workflow.remove_edges_from(new_edge)
            observation = nx.to_numpy_matrix(final_workflow).astype(int)
            if not self.observation_space.contains(observation):
                final_workflow.remove_edges_from(new_edge)
        return final_workflow

    def render(self, mode='human'):
        if mode == 'graph':
            # return graphtools graph object
            return self.graph
        elif mode == 'human':
            nx.draw(self.graph, with_labels=True, font_weight='bold')
            plt.show()

    def render_truth(self, mode='human'):
        if mode == 'graph':
            # return graphtools graph object
            return self.true_graph
        elif mode == 'human':
            nx.draw(self.true_graph, with_labels=True, font_weight='bold')
            plt.show()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: tuple):
        done = 0
        reward = 0
        assert self.action_space.contains(action)
        valid_source_nodes = [index for index, in_degree in
                              self.graph.in_degree() if
                              ((in_degree > 0 or index < self.input_nodes) and index < (self.network_size - 1))]

        if action[0] not in valid_source_nodes:
            self.time_step += 1
            return self.observation, reward, done, {"time_step": self.time_step}
            # raise ValueError('Action {} does not have a valid from node'.format(action))

        new_edge = [(action[0], action[1])]
        self.graph.add_edges_from(new_edge)

        if not nx.algorithms.dag.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edges_from(new_edge)
            self.time_step += 1
            return self.observation, reward, done, {"time_step": self.time_step}
            # raise ValueError('Action {} violates the DAG property'.format(action))

        self.observation = nx.to_numpy_matrix(self.graph).astype(int)

        if not self.observation_space.contains(self.observation):
            self.graph.remove_edges_from(new_edge)
            self.observation = nx.to_numpy_matrix(self.graph).astype(int)
            self.time_step += 1
            return self.observation, reward, done, {"time_step": self.time_step}
            # raise ValueError('Action {} makes a duplicate edge'.format(action))

        if nx.is_isomorphic(self.graph, self.true_graph):
            reward = 1
            done = 1

        self.time_step += 1

        return self.observation, reward, done, {"time_step": self.time_step}

    def reset(self):
        all_edges = list(self.graph.edges())
        self.graph.remove_edges_from(all_edges)
        self.time_step = 0
        self.observation = nx.to_numpy_matrix(self.graph).astype(int)
        return self.observation

    def get_action_size(self):
        """Gets the action_size for the gym env into the correct shape for a neural network"""
        return len(list(product(range(self.input_nodes), range(self.network_size))))

    def get_observation_size(self):
        return self.network_size ** 2
