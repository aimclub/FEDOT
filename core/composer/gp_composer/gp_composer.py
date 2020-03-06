from copy import deepcopy
from functools import partial
from typing import (
    List,
    Callable,
    Optional,
    SupportsInt,
    SupportsFloat
)

from core.composer.chain import Chain
from core.composer.composer import Composer, ComposerRequirements
from core.composer.gp_composer.gpnode import GPNode
from core.composer.gp_composer.gpnode import GPNodeGenerator
from core.composer.optimisers.gp_optimiser import GPChainOptimiser
from core.composer.visualisation import ComposerVisualiser
from core.models.data import InputData
from core.models.model import Model


class GPComposerRequirements(ComposerRequirements):
    def __init__(self, primary: List[Model], secondary: List[Model],
                 max_depth: Optional[SupportsInt], max_arity: Optional[SupportsInt], pop_size: Optional[SupportsInt],
                 num_of_generations: SupportsInt, crossover_prob: Optional[SupportsFloat],
                 mutation_prob: Optional[SupportsFloat] = None, verbose: bool = False, is_visualise: bool = False):
        super().__init__(primary=primary, secondary=secondary,
                         max_arity=max_arity, max_depth=max_depth, is_visualise=is_visualise)
        self.pop_size = pop_size
        self.num_of_generations = num_of_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.verbose = verbose


class GPComposer(Composer):
    def compose_chain(self, data: InputData, initial_chain: Optional[Chain],
                      composer_requirements: Optional[GPComposerRequirements],
                      metrics: Optional[Callable]) -> Chain:
        metric_function_for_nodes = partial(self._metric_for_nodes,
                                            metrics, data)
        optimiser = GPChainOptimiser(initial_chain=initial_chain,
                                     requirements=composer_requirements,
                                     primary_node_func=GPNodeGenerator.primary_node,
                                     secondary_node_func=GPNodeGenerator.secondary_node)

        best_found, history = optimiser.optimise(metric_function_for_nodes)

        if composer_requirements.is_visualise:
            historical_chains = []
            for historical_data in history:
                historical_nodes_set = GPComposer._tree_to_chain(historical_data[0], data).nodes
                historical_chain = Chain()
                [historical_chain.add_node(nodes) for nodes in historical_nodes_set]
                historical_chains.append(historical_chain)

        historical_fitnesses = [opt_step[1] for opt_step in history]

        if composer_requirements.is_visualise:
            ComposerVisualiser.visualise_history(historical_chains, historical_fitnesses)

        best_chain = GPComposer._tree_to_chain(tree_root=best_found, data=data)
        print("GP composition finished")
        return best_chain

    @staticmethod
    def _tree_to_chain(tree_root: GPNode, data: InputData) -> Chain:
        chain = Chain()
        nodes = GPComposer._flat_nodes_tree(deepcopy(tree_root))
        for node in nodes:
            if node.nodes_from:
                for i in range(len(node.nodes_from)):
                    node.nodes_from[i] = node.nodes_from[i]._chain_node
            chain.add_node(node._chain_node)
        chain.reference_data = data
        return chain

    @staticmethod
    def _flat_nodes_tree(node):
        if node.nodes_from:
            nodes = []
            for children in node.nodes_from:
                nodes += GPComposer._flat_nodes_tree(children)
            return [node] + nodes
        else:
            return [node]

    @staticmethod
    def _metric_for_nodes(metric_function, data, root: GPNode) -> float:
        chain = GPComposer._tree_to_chain(root, data)
        return metric_function(chain)
