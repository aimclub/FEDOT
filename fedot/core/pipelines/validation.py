from typing import Callable, List

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.dag.validation_rules import DEFAULT_DAG_RULES, has_no_cycle, has_no_isolated_nodes, \
    has_no_self_cycled_nodes, has_one_root
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.graph import OptGraph
from fedot.core.pipelines.validation_rules import has_correct_data_connections, has_correct_data_sources, \
    has_correct_operation_positions, has_final_operation_as_model, has_no_conflicts_in_decompose, \
    has_no_conflicts_with_data_flow, has_no_data_flow_conflicts_in_ts_pipeline, has_primary_nodes, \
    only_non_lagged_operations_are_primary, is_pipeline_contains_ts_operations

common_rules = [has_one_root,
                has_no_cycle,
                has_no_self_cycled_nodes,
                has_no_isolated_nodes,
                has_primary_nodes,
                has_correct_operation_positions,
                has_final_operation_as_model,
                has_no_conflicts_with_data_flow,
                has_no_conflicts_in_decompose,
                has_correct_data_connections,
                has_correct_data_sources]

ts_rules = [is_pipeline_contains_ts_operations,
            only_non_lagged_operations_are_primary,
            has_no_data_flow_conflicts_in_ts_pipeline]


def validate(graph: Graph, rules: List[Callable] = None, task=None):
    """ The graph is checked for compliance with the requirements

    :param graph: graph object
    :param rules: rules to check
    :param task: task which such a graph is solving
    """
    if not rules:
        rules = common_rules

    # Check common rules
    for rule_func in rules:
        _rule_check(graph, rule_func)

    # Perform time series specific rules
    if task and task.task_type is TaskTypesEnum.ts_forecasting:
        for rule_func in ts_rules:
            _rule_check(graph, rule_func)
    return True


def _rule_check(graph, rule_func):
    """ Perform graph check by rule """
    if rule_func in DEFAULT_DAG_RULES and isinstance(graph, OptGraph):
        graph = DirectAdapter(base_graph_class=Graph,
                              base_node_class=GraphNode).restore(graph)
    rule_func(graph)
