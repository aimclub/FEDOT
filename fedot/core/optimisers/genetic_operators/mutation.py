from copy import deepcopy
from functools import WRAPPER_ASSIGNMENTS
from random import choice
from typing import Dict, Callable

from golem.core.adapter import register_native
from golem.core.optimisers.genetic.operators.base_mutations import single_edge_mutation, single_add_mutation, \
    single_change_mutation, single_drop_mutation
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters


def _extract_graphs(graph: OptGraph) -> Dict[str, OptGraph]:
    """ Get all graphs from graph with atomized nodes
        Return dict with key as node uid (where graph is stored in atomized models)
        and values as graphs """
    graphs = {'': graph}
    for node in graph.nodes:
        if 'inner_graph' in node.content:
            extracted_graphs = _extract_graphs(node.content['inner_graph'])
            for k, v in extracted_graphs.items():
                graphs[k or node.uid] = v
    return graphs


def _insert_graphs(full_graph: OptGraph, node_uid: str, graph: OptGraph) -> OptGraph:
    """ Insert graph to full_graph with atomized model in node with uid node_uid """
    if node_uid == '':
        full_graph = graph
    else:
        full_graph = full_graph
        # look for node with uid == node_uid
        nodes = full_graph.nodes[:]
        while nodes:
            node = nodes.pop()
            if node.uid == node_uid:
                break
            if 'inner_graph' in node.content:
                nodes.extend(node.content['inner_graph'].nodes)
        else:
            raise ValueError(f"Unknown node uid: {node_uid}")
        if 'inner_graph' not in node.content:
            raise ValueError('Cannot insert graph to non AtomizedModel')
        node.content['inner_graph'] = graph
    return full_graph


MutationFun = Callable[[OptGraph, GraphRequirements, GraphGenerationParams, GPAlgorithmParameters], OptGraph]


def atomized_mutation(mutation_fun: MutationFun) -> MutationFun:
    def mutation_for_atomized_graph(graph: OptGraph,
                                    requirements: GraphRequirements,
                                    graph_gen_params: GraphGenerationParams,
                                    parameters: GPAlgorithmParameters,
                                    ) -> OptGraph:
        graph = deepcopy(graph)
        graphs = _extract_graphs(graph)
        node_uid, graph_to_mutate = choice(list(graphs.items()))

        mutated_graph = mutation_fun(graph=graph_to_mutate,
                                     requirements=requirements,
                                     graph_gen_params=graph_gen_params,
                                     parameters=parameters)

        new_graph = _insert_graphs(graph, node_uid, mutated_graph)
        return new_graph

    # TODO use functools.wraps. now it brokes something in GOLEM.
    for attr in WRAPPER_ASSIGNMENTS:
        setattr(mutation_for_atomized_graph, attr, getattr(mutation_fun, attr))
    mutation_for_atomized_graph.__wrapped__ = mutation_fun

    return register_native(mutation_for_atomized_graph)


fedot_single_edge_mutation = atomized_mutation(single_edge_mutation)
fedot_single_add_mutation = atomized_mutation(single_add_mutation)
fedot_single_change_mutation = atomized_mutation(single_change_mutation)
fedot_single_drop_mutation = atomized_mutation(single_drop_mutation)
