from typing import List, Optional

from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.utils import ComparableEnum as Enum


class RemoveType(Enum):
    node_only = 'node_only'
    with_direct_children = 'with_direct_children'
    with_parents = 'with_parents'
    forbidden = 'forbidden'


class DefaultChangeAdvisor:
    """
    Class for advising of pipeline changes during evolution
    """

    def __init__(self, task=None):
        self.task = task

    def propose_change(self, current_operation_id: str, possible_operations: List[str]):
        return possible_operations

    def can_be_removed(self, current_operation_id: str) -> RemoveType:
        return RemoveType.node_only

    def propose_parent(self, current_operation_id: str, parent_operations_ids: Optional[List[str]],
                       possible_operations: List[str]):
        return possible_operations


class PipelineChangeAdvisor(DefaultChangeAdvisor):
    def __init__(self, task=None):
        self.models = get_operations_for_task(task, mode='model')
        self.data_operations = get_operations_for_task(task, mode='data_operation')
        super().__init__(task)

    def can_be_removed(self, current_operation_id: str) -> RemoveType:
        if 'exog_ts' == current_operation_id:
            return RemoveType.forbidden
        if 'custom' in current_operation_id or 'lagged' in current_operation_id:
            return RemoveType.with_parents
        if 'data_source' in current_operation_id:
            return RemoveType.with_direct_children
        return RemoveType.node_only

    def propose_change(self, current_operation_id: str, possible_operations: List[str]):
        """
        Proposes promising candidates for node replacement
        :param current_operation_id: title of operation in current node
        :param possible_operations: list of candidates for replace
        :return:
        """
        if ('data_source' in current_operation_id or
                'exog_ts' == current_operation_id or
                'custom' in current_operation_id):
            # data source replacement is useless
            return [current_operation_id]

        is_model = current_operation_id in self.models
        similar_operations = self.models if is_model else self.data_operations

        candidates = set.intersection(set(similar_operations), set(possible_operations))

        if 'lagged' in current_operation_id:
            # lagged transform can be replaced only to lagged
            candidates = set.intersection({'lagged', 'sparse_lagged'}, set(possible_operations))

        if current_operation_id in candidates:
            # the change to the same node is not meaningful
            candidates.remove(current_operation_id)
        return list(candidates)

    def propose_parent(self, current_operation_id: str, parent_operations_ids: Optional[List[str]],
                       possible_operations: List[str]):
        """
        Proposes promising candidates for new parents
        :param current_operation_id: title of operation in current node
        :param parent_operations_ids: list of existing parents, None or [] if no parents
        :param possible_operations: list of candidates for replace
        :return:
        """

        candidates = set.intersection(set(self.data_operations), set(possible_operations))
        if current_operation_id in candidates:
            candidates.remove(current_operation_id)
        if parent_operations_ids:
            for parent_operation_id in parent_operations_ids:
                if parent_operation_id in candidates:
                    # the sequence of the same parent and child is not meaningful
                    candidates.remove(parent_operation_id)
        return list(candidates)
