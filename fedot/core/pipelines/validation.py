from typing import Callable, List

from fedot.core.pipelines.validation_rules import has_correct_operation_positions, has_final_operation_as_model, \
    has_no_conflicts_with_data_flow, has_correct_data_connections, has_no_data_flow_conflicts_in_ts_pipeline, \
    only_ts_specific_operations_are_primary, has_no_conflicts_in_decompose, has_primary_nodes
from fedot.core.dag.graph import Graph
from fedot.core.dag.validation_rules import has_one_root, has_no_cycle, has_no_isolated_nodes, has_no_self_cycled_nodes

default_rules = [has_one_root,
                 has_no_cycle,
                 has_no_self_cycled_nodes,
                 has_no_isolated_nodes,
                 has_primary_nodes,
                 has_correct_operation_positions,
                 has_final_operation_as_model,
                 has_no_conflicts_with_data_flow,
                 has_no_conflicts_in_decompose,
                 has_correct_data_connections,
                 only_ts_specific_operations_are_primary,
                 has_no_data_flow_conflicts_in_ts_pipeline]


def validate(graph: Graph, rules: List[Callable] = None):
    if not rules:
        rules = default_rules
    for rule_func in rules:
        rule_func(graph)

    return True
