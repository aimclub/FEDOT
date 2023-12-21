from copy import deepcopy
from functools import WRAPPER_ASSIGNMENTS
from random import choice

from typing import Callable, Union

from fedot.core.optimisers.genetic_operators.atomized_operators_wrapper import \
    extract_graphs_from_atomized, insert_graphs_to_atomized
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from golem.core.adapter import register_native
from golem.core.optimisers.genetic.operators.base_mutations import \
    single_edge_mutation, single_add_mutation, \
    single_change_mutation, single_drop_mutation, tree_growth
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters


OperationTypesRepository.init_repository('atomized')
ATOMIZED_OPERATION_REPOSITORY = OperationTypesRepository('atomized')

MutationFun = Callable[[OptGraph, GraphRequirements, GraphGenerationParams, GPAlgorithmParameters], OptGraph]


def atomized_mutation(mutation_fun: MutationFun) -> MutationFun:
    def mutation_for_atomized_graph(graph: OptGraph,
                                    requirements: GraphRequirements,
                                    graph_gen_params: GraphGenerationParams,
                                    parameters: GPAlgorithmParameters,
                                    ) -> OptGraph:
        graph = deepcopy(graph)
        graphs = extract_graphs_from_atomized(graph)
        node_uid, graph_to_mutate = choice(list(graphs.items()))

        mutated_graph = mutation_fun(graph_to_mutate,
                                     requirements=requirements,
                                     graph_gen_params=graph_gen_params,
                                     parameters=parameters)

        new_graph = insert_graphs_to_atomized(graph, node_uid, mutated_graph)
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
fedot_tree_growth = register_native(atomized_mutation(tree_growth))


@atomized_mutation
def insert_atomized_operation(pipeline: Pipeline,
                              requirements: GraphRequirements,
                              graph_gen_params: GraphGenerationParams,
                              parameters: GPAlgorithmParameters,
                              ) -> Pipeline:
    """ Wrap part of pipeline to atomized operation
    """
    task_type = graph_gen_params.advisor.task.task_type
    atomized_operations = ATOMIZED_OPERATION_REPOSITORY.suitable_operation(task_type=task_type, tags=['non-default'])
    atomized_operation = choice(atomized_operations)
    atomized_operation = 'atomized_ts_differ'

    info = ATOMIZED_OPERATION_REPOSITORY.operation_info_by_id(atomized_operation)
    it, ot = set(info.input_types), set(info.output_types)

    nodes = list()
    for node in pipeline.nodes:
        if (set(node.operation.metadata.input_types) == it and
            set(node.operation.metadata.output_types) == ot):
            nodes.append(node)

    if nodes:
        node = choice(nodes)
        inner_pipeline = Pipeline(PipelineNode(content=node.content))

        # create new node_factory and graph_random_factory for new pipeline
        strategy = info.current_strategy(graph_gen_params.advisor.task)
        operation_class = strategy._operations_by_types[atomized_operation]
        node_factory, graph_random_factory = operation_class.build_factories(requirements, graph_gen_params)
        inner_pipeline.node_factory = node_factory
        inner_pipeline.graph_random_factory = graph_random_factory

        # build new node with inner pipeline
        new_node = PipelineNode(content={'name': atomized_operation, 'params': {'pipeline': inner_pipeline}})
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
