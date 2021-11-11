from copy import copy

from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.tasks import Task


class OperationsPreset:
    """ Class for presets processing. Preset is a set of operations (data operations
    and models), which will be used during pipeline structure search
    """

    def __init__(self, task: Task, preset_name: str):
        self.task = task
        self.preset_name = preset_name

    def composer_params_based_on_preset(self, composer_params: dict) -> dict:
        """ Return composer parameters dictionary with appropriate operations
        based on defined preset
        """
        updated_params = copy(composer_params)

        if self.preset_name is None and 'preset' in updated_params:
            self.preset_name = updated_params['preset']

        if 'preset' in updated_params:
            del updated_params['preset']

        if self.preset_name is not None:
            available_operations = self._filter_operations_by_preset()
            updated_params['available_operations'] = available_operations

        if updated_params['with_tuning'] or '_tun' in self.preset_name:
            updated_params['with_tuning'] = True

        return updated_params

    def _filter_operations_by_preset(self):
        """ Filter operations by preset, remove "heavy" operations and save
        appropriate ones
        """
        excluded = ['mlp', 'svc', 'svr', 'arima', 'exog_ts', 'text_clean']

        # TODO remove workaround
        extended_excluded = ['mlp', 'catboost', 'lda', 'qda', 'lgbm',
                             'svc', 'svr', 'arima', 'exog_ts', 'text_clean',
                             'one_hot_encoding']
        excluded_models_dict = {'light': excluded,
                                'light_tun': excluded,
                                'light_steady_state': extended_excluded}

        # Get data operations and models
        available_operations = get_operations_for_task(self.task, mode='all')
        available_data_operation = get_operations_for_task(self.task, mode='data_operation')

        # Exclude "heavy" operations if necessary
        available_operations = self._new_operations_without_heavy(excluded_models_dict, available_operations)

        # Save only "light" operations
        if self.preset_name in ['ultra_light', 'ultra_light_tun', 'ultra_steady_state']:
            light_models = ['dt', 'dtreg', 'logit', 'linear', 'lasso', 'ridge', 'knn', 'ar']
            included_operations = light_models + available_data_operation
            available_operations = [_ for _ in available_operations if _ in included_operations]
        elif self.preset_name in ['ts', 'ts_tun']:
            # Presets for time series forecasting
            available_operations = ['lagged', 'sparse_lagged', 'ar', 'gaussian_filter', 'smoothing',
                                    'ridge', 'linear', 'lasso', 'dtreg', 'scaling', 'normalization',
                                    'pca']

        if self.preset_name == 'gpu':
            repository = OperationTypesRepository().assign_repo('model', 'gpu_models_repository.json')
            available_operations = repository.suitable_operation(task_type=self.task.task_type)

        return available_operations

    def _new_operations_without_heavy(self, excluded_models_dict, available_operations) -> list:
        """ Create new list without heavy operations """
        if self.preset_name in excluded_models_dict.keys():
            excluded_operations = excluded_models_dict[self.preset_name]
            available_operations = [_ for _ in available_operations if _ not in excluded_operations]

        return available_operations

