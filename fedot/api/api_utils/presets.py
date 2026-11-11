from copy import copy
from typing import Optional

from fedot.api.api_utils.assumptions.assumption_rules import (
    exclude_operations,
    finalize_operations,
    merge_preset_operations,
    parse_preset_spec,
)
from fedot.api.time import ApiTime
from fedot.core.operations.operation_execution import filter_available_operations
from fedot.core.constants import BEST_QUALITY_PRESET_NAME, \
    FAST_TRAIN_PRESET_NAME, AUTO_PRESET_NAME
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.tasks import Task


class OperationsPreset:
    """ Class for presets processing. Preset is a set of operations (data operations
    and models), which will be used during pipeline structure search
    """

    def __init__(self, task: Task, preset_name: str):
        self.task = task
        self.preset_name = preset_name

        # Is there a modification in preset or not
        self.modification_using = False

    def composer_params_based_on_preset(self, api_params: dict, data_type: Optional[DataTypesEnum] = None,
                                        data_device=None, multi_output: bool = False) -> dict:
        """ Return composer parameters dictionary with appropriate operations
        based on defined preset
        """
        updated_params = copy(api_params)

        if self.preset_name is None and 'preset' in updated_params:
            self.preset_name = updated_params['preset']

        if self.preset_name is not None and api_params.get('available_operations') is None:
            available_operations = self.filter_operations_by_preset(data_type)
            updated_params['available_operations'] = available_operations
        elif self.preset_name is not None:
            parse_preset_spec(self.preset_name)

        if data_device is not None and updated_params.get('available_operations') is not None:
            updated_params['available_operations'] = self.filter_available_operations(
                updated_params['available_operations'], data_device, multi_output)

        return updated_params

    def filter_operations_by_preset(self, data_type: Optional[DataTypesEnum] = None,
                                    data_device=None, multi_output: bool = False):
        """ Filter operations by preset, remove "heavy" operations and save
        appropriate ones
        """
        preset_spec = parse_preset_spec(self.preset_name)

        if preset_spec.use_auto:
            available_operations = get_operations_for_task(
                self.task, data_type, mode='all')
            return (self.filter_available_operations(available_operations, data_device, multi_output)
                    if data_device is not None else available_operations)

        excluded = ['mlp', 'svc', 'svr', 'arima', 'exog_ts', 'text_clean',
                    'lda', 'qda', 'lgbm', 'polyfit',
                    'resample', 'stl_arima']
        excluded_tree = []

        self.modification_using = preset_spec.modification is not None
        base_operations = get_operations_for_task(
            self.task,
            data_type,
            mode='all',
            preset=preset_spec.base_preset,
        )
        if self.modification_using:
            mod_operations = get_operations_for_task(
                self.task,
                data_type,
                mode='all',
                preset=preset_spec.modification,
            )
            available_operations = list(merge_preset_operations(base_operations, mod_operations))
        else:
            available_operations = base_operations

        if preset_spec.use_stable:
            available_operations = list(
                exclude_operations(available_operations, excluded))

        available_operations = finalize_operations(available_operations, excluded_tree)
        return (self.filter_available_operations(available_operations, data_device, multi_output)
                if data_device is not None else available_operations)

    def filter_available_operations(self, operation_names, data_device, multi_output: bool = False):
        repository = OperationTypesRepository('all')
        metadata = {operation.id: operation for operation in repository.operations}
        known = [metadata[name] for name in operation_names if name in metadata]
        available = {operation.id for operation in filter_available_operations(
            known, self.task.task_type, data_device, multi_output)}
        # Extension operations have their own capabilities; keep their existing query path.
        result = [name for name in operation_names if name not in metadata or name in available]
        if operation_names and not result:
            raise ValueError(f'No operations are available for TensorData on {data_device}.')
        return result

    @staticmethod
    def new_operations_without_heavy(excluded_operations, available_operations) -> list:
        """ Create new list without heavy operations """
        return list(exclude_operations(available_operations, excluded_operations))


def change_preset_based_on_initial_fit(timer: ApiTime, n_jobs: int) -> str:
    """
    If preset was set as 'auto', based on initial pipeline fit time, appropriate one can be chosen
    """
    if timer.time_for_automl in [-1, None]:
        return BEST_QUALITY_PRESET_NAME

    # Change preset to appropriate one

    if timer.have_time_for_the_best_quality(n_jobs=n_jobs):
        # It is possible to train only few number of pipelines during optimization - use simplified preset
        return BEST_QUALITY_PRESET_NAME
    else:
        return FAST_TRAIN_PRESET_NAME
