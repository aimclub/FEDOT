from typing import Callable, List, Sequence, Optional, Union

from fedot.core.dag.graph import Graph
from fedot.core.dag.validation_rules import (
    DEFAULT_DAG_RULES,
    has_no_cycle,
    has_no_isolated_nodes,
    has_no_self_cycled_nodes,
    has_one_root
)
from fedot.core.log import Log, default_log
from fedot.core.optimisers.adapters import DirectAdapter, BaseOptimizationAdapter, PipelineAdapter
from fedot.core.optimisers.graph import OptGraph
from fedot.core.pipelines.validation_rules import (
    has_correct_data_connections,
    has_correct_data_sources,
    has_correct_operation_positions,
    has_final_operation_as_model,
    has_no_conflicts_after_class_decompose,
    has_no_conflicts_during_multitask,
    has_no_conflicts_in_decompose,
    has_no_conflicts_with_data_flow,
    has_no_data_flow_conflicts_in_ts_pipeline,
    has_primary_nodes,
    is_pipeline_contains_ts_operations,
    only_non_lagged_operations_are_primary
)
from fedot.core.repository.tasks import TaskTypesEnum

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

class_rules = [has_no_conflicts_during_multitask,
               has_no_conflicts_after_class_decompose]


# Validation rule can either return False or raise a ValueError to signal a failed check
ValidateRuleType = Callable[..., bool]


def rules_by_task(task_type: Optional[TaskTypesEnum], rules: Sequence[ValidateRuleType] = ()) -> Sequence[ValidateRuleType]:
    tmp_rules = []

    tmp_rules.extend(rules or common_rules)

    if task_type is TaskTypesEnum.ts_forecasting:
        tmp_rules.extend(ts_rules)
    elif task_type is TaskTypesEnum.classification:
        tmp_rules.extend(class_rules)

    return tmp_rules


class GraphValidator:
    def __init__(self,
                 rules: Sequence[ValidateRuleType] = (),
                 adapter: Optional[BaseOptimizationAdapter] = None,
                 log: Optional[Log] = None):
        self._rules = rules
        self._adapter = adapter or DirectAdapter()
        self._log = log or default_log(self.__class__.__name__)

    @staticmethod
    def for_task(task_type: Optional[TaskTypesEnum] = None,
                 adapter: Optional[BaseOptimizationAdapter] = None,
                 log: Optional[Log] = None):
        return GraphValidator(rules_by_task(task_type), adapter, log)

    def __call__(self, graph: Union[Graph, OptGraph]) -> bool:
        return self.validate(graph)

    def validate(self, graph: Union[Graph, OptGraph]) -> bool:
        restored_graph: Graph = self._adapter.restore(graph)
        # Check if all rules pass
        for rule in self._rules:
            try:
                if rule(restored_graph) is False:
                    return False
            except ValueError as err:
                self._log.info(f'Graph validation failed with error <{err}> '
                               f'for rule={rule} on graph={restored_graph.root_node.descriptive_id}.')
                return False
        return True


def validate_pipeline(graph: Union[Graph, OptGraph], rules: List[Callable] = None, task_type: Optional[TaskTypesEnum] = None):
    """Method for validation of graphs with default rules.
    NB: It is preserved for simplicity, use GraphValidator instead."""
    adapter = PipelineAdapter()
    return GraphValidator(rules_by_task(task_type, rules), adapter).validate(graph)
