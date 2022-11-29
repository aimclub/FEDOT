from typing import List

from fedot.core.optimisers.advisor import DefaultChangeAdvisor, RemoveType
from fedot.core.optimisers.graph import OptNode
from fedot.core.repository.operation_types_repository import get_operations_for_task


class PipelineChangeAdvisor(DefaultChangeAdvisor):
    def __init__(self, task=None):
        self.models: List[str] = get_operations_for_task(task, mode='model')
        self.data_operations: List[str] = get_operations_for_task(task, mode='data_operation')
        super().__init__(task)

    def can_be_removed(self, node: OptNode) -> RemoveType:
        operation_id = node.content['name']
        if 'exog_ts' in operation_id:
            return RemoveType.forbidden
        if 'custom' in operation_id or 'lagged' in operation_id:
            return RemoveType.with_parents
        if 'data_source' in operation_id:
            return RemoveType.with_direct_children
        return RemoveType.node_only

    def propose_change(self, node: OptNode, possible_operations: List[str]) -> List[str]:
        """
        Proposes promising candidates for node replacement
        :param node: node to propose changes for
        :param possible_operations: list of candidates for replace
        :return: list of candidates with str operations
        """
        operation_id = node.content['name']
        # data source, exog_ts and custom models replacement is useless
        if check_for_specific_operations(operation_id):
            return []

        is_model = operation_id in self.models
        similar_operations = self.models if is_model else self.data_operations

        candidates = set.intersection(set(similar_operations), set(possible_operations))

        if 'lagged' in operation_id:
            # lagged transform can be replaced only to lagged
            candidates = set.intersection({'lagged', 'sparse_lagged'}, set(possible_operations))

        if operation_id in candidates:
            # the change to the same node is not meaningful
            candidates.remove(operation_id)
        return candidates

    def propose_parent(self, node: OptNode, possible_operations: List[str]) -> List[str]:
        """
        Proposes promising candidates for new parents
        :param node: node to propose changes for
        :param possible_operations: list of candidates for replace
        :return: list of candidates with str operations
        """

        operation_id = node.content['name']
        if check_for_specific_operations(operation_id):
            # data source, exog_ts and custom models moving is useless
            return []

        parent_operations = [str(n.content['name']) for n in node.nodes_from]
        candidates = set.intersection(set(self.data_operations), set(possible_operations))

        if operation_id in candidates:
            candidates.remove(operation_id)
        if parent_operations:
            for parent_operation_id in parent_operations:
                if check_for_specific_operations(parent_operation_id):
                    # data source, exog_ts and custom models moving is useless
                    return []
                if parent_operation_id in candidates:
                    # the sequence of the same parent and child is not meaningful
                    candidates.remove(parent_operation_id)
        return candidates


def check_for_specific_operations(operation_id: str):
    if ('data_source' in operation_id or
            'exog_ts' == operation_id or 'custom' in operation_id):
        return True
    return False
