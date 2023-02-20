from copy import deepcopy
from random import randint
from typing import Optional

from golem.core.constants import MAX_GRAPH_GEN_ATTEMPTS
from golem.core.dag.graph_utils import distance_to_root_level
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.random_graph_factory import RandomGraphFactory

from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory


class RandomPipelineFactory(RandomGraphFactory):
    """ Default realisation of random graph factory. Generates DAG graph using random growth. """

    def __init__(self,
                 verifier: GraphVerifier,
                 node_factory: PipelineOptNodeFactory):
        self.node_factory = node_factory
        self.verifier = verifier

    def __call__(self, requirements: PipelineComposerRequirements, max_depth: Optional[int] = None) -> OptGraph:
        return random_pipeline(self.verifier, self.node_factory, requirements, max_depth)


def random_pipeline(verifier: GraphVerifier,
                    node_factory: PipelineOptNodeFactory,
                    requirements: PipelineComposerRequirements,
                    max_depth: Optional[int] = None) -> OptGraph:
    max_depth = max_depth if max_depth else requirements.max_depth
    is_correct_graph = False
    graph = None
    n_iter = 0
    requirements = adjust_requirements(requirements)

    while not is_correct_graph:
        graph = OptGraph()
        if requirements.max_depth == 1:
            graph_root = node_factory.get_node(is_primary=True)
            graph.add_node(graph_root)
        else:
            graph_root = node_factory.get_node(is_primary=False)
            graph.add_node(graph_root)
            graph_growth(graph, graph_root, node_factory, requirements, max_depth)

        is_correct_graph = verifier(graph)
        n_iter += 1
        if n_iter > MAX_GRAPH_GEN_ATTEMPTS:
            raise ValueError(f'Could not generate random graph for {n_iter} '
                             f'iterations with requirements {requirements}')
    return graph


def adjust_requirements(requirements: PipelineComposerRequirements) -> PipelineComposerRequirements:
    """Function returns modified copy of the requirements if necessary.
    Example: Graph with only one primary node should consist of only one primary node
    without duplication, because this causes errors. Therefore minimum and maximum arity
    become equal to one.
    """
    requirements = deepcopy(requirements)
    if len(requirements.primary) == 1 and requirements.max_arity > 1:
        requirements.min_arity = requirements.max_arity = 1
    return requirements


def graph_growth(graph: OptGraph,
                 node_parent: OptNode,
                 node_factory: PipelineOptNodeFactory,
                 requirements: PipelineComposerRequirements,
                 max_depth: int):
    """Function create a graph and links between nodes"""
    offspring_size = randint(requirements.min_arity, requirements.max_arity)

    for offspring_node in range(offspring_size):
        height = distance_to_root_level(graph, node_parent)
        is_max_depth_exceeded = height >= max_depth - 2
        is_primary_node_selected = height < max_depth - 1 and randint(0, 1)
        if is_max_depth_exceeded or is_primary_node_selected:
            primary_node = node_factory.get_node(is_primary=True)
            node_parent.nodes_from.append(primary_node)
            graph.add_node(primary_node)
        else:
            secondary_node = node_factory.get_node(is_primary=False)
            graph.add_node(secondary_node)
            node_parent.nodes_from.append(secondary_node)
            graph_growth(graph, secondary_node, node_factory, requirements, max_depth)