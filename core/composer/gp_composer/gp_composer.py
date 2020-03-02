from core.composer.composer import Composer, ComposerRequirements
from typing import (
    List,
    Callable,
    Optional,
    SupportsInt
)

from core.composer.chain import Chain
from functools import partial

from core.models.model import Model
from core.models.data import InputData
from core.composer.gp_composer.optimisers.gp_optimiser import GPChainOptimiser
from core.composer.gp_composer.gp_node import GP_NodeGenerator
from core.composer.gp_composer.gp_node import GP_Node
from copy import deepcopy


class GPComposer_requirements(ComposerRequirements):
    def __init__(self, primary_requirements: List[Model], secondary_requirements: List[Model],
                 max_depth: Optional[SupportsInt], max_arity: Optional[SupportsInt], pop_size: Optional[SupportsInt],
                 num_of_generations: SupportsInt, minimization=False):
        super().__init__(primary_requirements=primary_requirements, secondary_requirements=secondary_requirements,
                         max_arity=max_arity, max_depth=max_depth)
        self.pop_size = pop_size
        self.num_of_generations = num_of_generations
        self.minimization =minimization


class GPComposer(Composer):
    def compose_chain(self, data: InputData, initial_chain: Optional[Chain],
                      composer_requirements: Optional[GPComposer_requirements],
                      metrics: Optional[Callable]) -> Chain:
        metric_function_for_nodes = partial(self._metric_for_nodes,
                                            metrics, data)
        optimiser = GPChainOptimiser(initial_chain=initial_chain,
                                     requirements=composer_requirements,
                                     primary_node_func=GP_NodeGenerator.primary_node,
                                     secondary_node_func=GP_NodeGenerator.secondary_node)
        best_chain = GPComposer._tree_to_chain(tree_root=optimiser.optimise(metric_function_for_nodes),data=data)
        return best_chain

    @staticmethod
    def _tree_to_chain(tree_root: GP_Node, data: InputData) -> Chain:
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
    def _metric_for_nodes(metric_function, data, root: GP_Node) -> float:
        chain = GPComposer._tree_to_chain(root, data)
        return metric_function(chain)
