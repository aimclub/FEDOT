import itertools
from typing import List, Optional, Dict

import numpy as np

from fedot.api.api_utils.presets import OperationsPreset
from fedot.core.repository.graph_operation_repository import GraphOperationRepository
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum


class PipelineOperationRepository(GraphOperationRepository):
    """Repository in order to extract suitable operations by keys
     for pipelines during graph composition. Defines 2 keys:
     `primary` and `secondary` for distinguishing operations suitable
     for, respectively, primary and secondary nodes.

     Designed to work in cooperation with
     :py:class:`fedot.core.optimisers.opt_node_factory.OptNodeFactory`

     Args:
        operations_by_keys: available operations already splitted by keys
     """
    def __init__(self, operations_by_keys: Optional[Dict[str, List[str]]] = None):
        super().__init__()
        self.operations_by_keys = operations_by_keys or dict()

    def from_available_operations(self, task: Task, preset: str,
                                  available_operations: List[str]):
        """ Initialize repository from available operations, task and preset """
        operations_by_task_preset = OperationsPreset(task, preset).filter_operations_by_preset()
        all_operations = list(set.intersection(set(operations_by_task_preset), set(available_operations)))
        primary_operations, secondary_operations = \
            self.divide_operations(all_operations, task)
        self.operations_by_keys = {'primary': primary_operations, 'secondary': secondary_operations}

    def get_operations(self, is_primary: bool) -> List[str]:
        """ Get pipeline operations by specified model key """
        operation_key = 'primary' if is_primary else 'secondary'
        operations = self.operations_by_keys.get(operation_key, [])
        return operations

    def get_all_operations(self) -> List[str]:
        """ Get all pipeline operations with all keys """
        return list(itertools.chain(*self.operations_by_keys.values()))

    @staticmethod
    def divide_operations(available_operations, task):
        """ Function divide operations for primary and secondary """

        if task.task_type is TaskTypesEnum.ts_forecasting:
            # Get time series operations for primary nodes
            ts_data_operations = get_operations_for_task(task=task,
                                                         mode='data_operation',
                                                         tags=["non_lagged"])
            # Remove exog data operation from the list
            ts_data_operations.remove('exog_ts')

            ts_primary_models = get_operations_for_task(task=task,
                                                        mode='model',
                                                        tags=["non_lagged"])
            # Union of the models and data operations
            ts_primary_operations = ts_data_operations + ts_primary_models

            # Filter - remain only operations, which were in available ones
            primary_operations = list(set(ts_primary_operations).intersection(available_operations))
            secondary_operations = available_operations
        else:
            primary_operations = available_operations
            secondary_operations = available_operations
        return primary_operations, secondary_operations
