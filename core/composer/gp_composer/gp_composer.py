from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import (
    Callable,
    Optional,
    List
)

from core.chain_validation import validate
from core.composer.chain import Chain, SharedChain
from core.composer.composer import Composer, ComposerRequirements
from core.composer.node import NodeGenerator
from core.composer.optimisers.gp_node import GPNode
from core.composer.optimisers.gp_optimiser import GPChainOptimiser
from core.composer.visualisation import ComposerVisualiser
from core.models.data import InputData
from core.models.data import train_test_data_setup


@dataclass
class GPComposerRequirements(ComposerRequirements):
    pop_size: Optional[int] = 50
    num_of_generations: Optional[int] = 50
    crossover_prob: Optional[float] = None
    mutation_prob: Optional[float] = None


class GPComposer(Composer):
    def __init__(self):
        super(Composer, self).__init__()
        self.shared_cache = {}

    def compose_chain(self, data: InputData, initial_chain: Optional[Chain],
                      composer_requirements: Optional[GPComposerRequirements],
                      metrics: Optional[Callable], is_visualise: bool = False) -> Chain:

        train_data, test_data = train_test_data_setup(data, 0.8)
        self.shared_cache.clear()

        metric_function_for_nodes = partial(self.metric_for_nodes,
                                            metrics, train_data, test_data)

        optimiser = GPChainOptimiser(initial_chain=initial_chain,
                                     requirements=composer_requirements,
                                     primary_node_func=NodeGenerator.primary_node,
                                     secondary_node_func=NodeGenerator.secondary_node)

        best_found, history = optimiser.optimise(metric_function_for_nodes)

        historical_chains = []
        for historical_data in history:
            historical_chain = self.tree_to_chain(historical_data[0])
            historical_chains.append(historical_chain)
        historical_fitness = [opt_step[1] for opt_step in history]

        self.history = [(chain, fitness) for chain, fitness in zip(historical_chains, historical_fitness)]

        if is_visualise:
            ComposerVisualiser.visualise_history(historical_chains, historical_fitness)

        best_chain = self.tree_to_chain(tree_root=best_found)
        print("GP composition finished")
        return best_chain

    def tree_to_chain(self, tree_root: GPNode, is_chain_shared: bool = False) -> Chain:
        chain = Chain()

        if is_chain_shared:
            chain = SharedChain(base_chain=chain, shared_cache=self.shared_cache)

        nodes = flat_nodes_tree(deepcopy(tree_root))
        for node in nodes:
            if node.nodes_from:
                for i in range(len(node.nodes_from)):
                    node.nodes_from[i] = node.nodes_from[i].chain_node
            chain.add_node(node.chain_node)
        validate(chain)
        return chain

    def metric_for_nodes(self, metric_function, train_data: InputData,
                         test_data: InputData,
                         root: GPNode) -> float:
        chain = self.tree_to_chain(root, is_chain_shared=True)
        chain.fit(input_data=train_data)
        return metric_function(chain, test_data)


def flat_nodes_tree(node: GPNode) -> List[GPNode]:
    if node.nodes_from:
        nodes = []
        for children in node.nodes_from:
            nodes += flat_nodes_tree(children)
        return [node] + nodes
    else:
        return [node]
