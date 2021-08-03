from typing import List, Optional

from fedot.core.repository.operation_types_repository import get_operations_for_task


class DefaultChangeAdvisor:
    """
    Class for advising of pipeline changes during evolution
    """

    def __init__(self, task=None):
        self.task = task

    def propose_change(self, current_operation_id: str, possible_operations: List[str]):
        return possible_operations

    def propose_parent(self, current_operation_id: str, parent_operations_ids: Optional[List[str]],
                       possible_operations: List[str]):
        return possible_operations


class PipelineChangeAdvisor(DefaultChangeAdvisor):
    def __init__(self, task=None):
        self.models = get_operations_for_task(task, mode='model')
        self.data_operations = get_operations_for_task(task, mode='data_operation')
        super().__init__(task)

    def propose_change(self, current_operation_id: str, possible_operations: List[str]):
        """
        Proposes promising candidates for node replacement
        :param current_operation_id: title of operation in current node
        :param possible_operations: list of candidates for replace
        :return:
        """
        if 'data_source' in current_operation_id:
            # data source replacement is useless
            return current_operation_id
        is_model = current_operation_id in self.models
        if is_model:
            candidates = set.intersection(set(self.models), set(possible_operations))
        else:
            candidates = set.intersection(set(self.data_operations), set(possible_operations))
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
