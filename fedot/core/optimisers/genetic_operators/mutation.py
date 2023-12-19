from copy import deepcopy
from functools import partial, wraps, WRAPPER_ASSIGNMENTS
from itertools import chain
from random import choice, randint, random, sample
from typing import TYPE_CHECKING, Optional, Dict, Callable, Union

from fedot.core.operations.atomized_model.atomized_model import AtomizedModel
from fedot.core.operations.atomized_model.atomized_ts_differ import AtomizedTimeSeriesDiffer
from fedot.core.operations.atomized_model.atomized_ts_sampler import AtomizedTimeSeriesSampler
from fedot.core.operations.atomized_model.atomized_ts_scaler import AtomizedTimeSeriesScaler
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import get_operations_for_task, OperationTypesRepository
from golem.core.adapter import register_native
from golem.core.dag.graph import ReconnectType
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.graph_utils import distance_to_root_level, distance_to_primary_level, graph_has_cycle
from golem.core.optimisers.advisor import RemoveType
from golem.core.optimisers.genetic.operators.base_mutations import \
    add_as_child, add_separate_parent_node, add_intermediate_node, single_edge_mutation, single_add_mutation, \
    single_change_mutation, single_drop_mutation
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters
from golem.utilities.data_structures import ComparableEnum as Enum
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters



OperationTypesRepository.init_repository('atomized')
ATOMIZED_OPERATION_REPOSITORY = OperationTypesRepository('atomized')
# TODO add ability to construct PipelineNodes as PipelineNode(name_of_atomized_operation)
ATOMIZED_OPERATION_MAP = {'atomized_ts_differ': AtomizedTimeSeriesDiffer,
                          'atomized_ts_scaler': AtomizedTimeSeriesScaler,
                          'atomized_ts_sampler': AtomizedTimeSeriesSampler}


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
            raise ValueError(f"Cannot insert graph to non AtomizedModel")
        node.content['inner_graph'] = graph
    return full_graph


MutationFun = Callable[[Union[OptGraph, Pipeline], GraphRequirements, GraphGenerationParams, GPAlgorithmParameters],
                       Union[OptGraph, Pipeline]]


def atomized_mutation(mutation_fun: MutationFun) -> MutationFun:
    def mutation_for_atomized_graph(graph: Union[OptGraph, Pipeline],
                                    requirements: GraphRequirements,
                                    graph_gen_params: GraphGenerationParams,
                                    parameters: GPAlgorithmParameters,
                                    ) -> Union[OptGraph, Pipeline]:
        graph = deepcopy(graph)
        graphs = _extract_graphs(graph)
        node_uid, graph_to_mutate = choice(list(graphs.items()))

        mutated_graph = mutation_fun(graph_to_mutate,
                                     requirements=requirements,
                                     graph_gen_params=graph_gen_params,
                                     parameters=parameters)

        new_graph = _insert_graphs(graph, node_uid, mutated_graph)
        return new_graph

    # TODO use functools.wraps. now it brokes something in GOLEM.
    for attr in WRAPPER_ASSIGNMENTS:
        setattr(mutation_for_atomized_graph, attr, getattr(mutation_fun, attr))
    mutation_for_atomized_graph.__wrapped__ = mutation_fun

    return mutation_for_atomized_graph


fedot_single_edge_mutation = register_native(atomized_mutation(single_edge_mutation))
fedot_single_add_mutation = register_native(atomized_mutation(single_add_mutation))
fedot_single_change_mutation = register_native(atomized_mutation(single_change_mutation))
fedot_single_drop_mutation = register_native(atomized_mutation(single_drop_mutation))


# TODO make insert_ts_atomized_operation atomized mutation
# @atomized_mutation
def insert_atomized_operation(pipeline: Pipeline,
                              requirements: GraphRequirements,
                              graph_gen_params: GraphGenerationParams,
                              parameters: GPAlgorithmParameters,
                              ) -> Pipeline:
    """ Wrap part of pipeline to atomized operation
    """
    task_type = graph_gen_params.advisor.task.task_type
    atomized_operations = ATOMIZED_OPERATION_REPOSITORY.suitable_operation(task_type=task_type)
    atomized_operation = choice(atomized_operations)
    info = ATOMIZED_OPERATION_REPOSITORY.operation_info_by_id(atomized_operation)
    it, ot = set(info.input_types), set(info.output_types)

    nodes = list()
    for node in pipeline.nodes:
        if (set(node.operation.metadata.input_types) == it and
            set(node.operation.metadata.output_types) == ot):
            nodes.append(node)

    if nodes:
        node = choice(nodes)
        new_node = PipelineNode(ATOMIZED_OPERATION_MAP[atomized_operation](Pipeline(node)))
        pipeline.update_node(node, new_node)
    return pipeline


# # idea is to get any part of graph and put it to new operation
# # TODO refactor and make more common algorithm
# # TODO adapt algorithm for high depth graphs
# root_nodes = pipeline.root_nodes()
# if len(root_nodes) != 1:
#     raise ValueError('mutation works with the only root node')
#
# allowed_operations = get_operations_for_task(None, mode='model')
# nodes_to_wrap = list()
# while not nodes_to_wrap:
#     nodes = deepcopy(root_nodes)
#     while nodes:
#         node = nodes.pop()
#         next_nodes = list()
#         for _node in node.nodes_from:
#             if random() > 0.3:
#                 next_nodes.append(deepcopy(_node))
#         if random() > 0.5:
#             node.nodes_from = next_nodes
#             nodes_to_wrap.append(node)
#             nodes.extend(next_nodes)
# pipeline = Pipeline(nodes_to_wrap[0])
