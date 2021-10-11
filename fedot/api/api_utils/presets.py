from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task


class OperationPreset:
    """ Class for presets processing. Preset is a set of operations (data operations
    and models), which will be used during pipeline structure search
    """

    def __init__(self, task, preset: str):
        self.task = task
        self.preset = preset

    def get_preset(self, composer_params: dict):
        """ Return composer parameters dictionary with appropriate operations
        based on defined preset
        """
        if self.preset is None and 'preset' in composer_params:
            self.preset = composer_params['preset']

        if 'preset' in composer_params:
            del composer_params['preset']

        if self.preset is not None:
            available_operations = self._filter_operations_by_preset()
            composer_params['available_operations'] = available_operations

        if composer_params['with_tuning'] or '_tun' in self.preset:
            composer_params['with_tuning'] = True

        return composer_params

    def _filter_operations_by_preset(self):
        """ Filter operations by preset, remove "heavy" operations and save
        appropriate ones
        """
        excluded = ['mlp', 'svc', 'svr', 'arima', 'exog_ts', 'text_clean']

        # TODO remove workaround
        extended_excluded = ['mlp', 'catboost', 'lda', 'qda', 'lgbm',
                             'svc', 'svr', 'arima', 'exog_ts', 'text_clean',
                             'one_hot_encoding', 'ransac_lin_reg', 'ransac_non_lin_reg']
        excluded_models_dict = {'light': excluded,
                                'light_tun': excluded,
                                'light_steady_state': extended_excluded}

        # Get data operations and models
        available_operations = get_operations_for_task(self.task, mode='all')
        available_data_operation = get_operations_for_task(self.task, mode='data_operation')

        # Exclude "heavy" operations if necessary
        available_operations = self._remove_heavy_operations(excluded_models_dict, available_operations)

        # Save only "light" operations
        if self.preset in ['ultra_light', 'ultra_light_tun', 'ultra_steady_state']:
            light_models = ['dt', 'dtreg', 'logit', 'linear', 'lasso', 'ridge', 'knn', 'ar']
            included_operations = light_models + available_data_operation
            available_operations = [_ for _ in available_operations if _ in included_operations]
        elif self.preset in ['ts', 'ts_tun']:
            # Presets for time series forecasting
            available_operations = ['lagged', 'sparse_lagged', 'ar', 'gaussian_filter', 'smoothing',
                                    'ridge', 'linear', 'lasso', 'dtreg', 'decompose']

        if self.preset == 'gpu':
            # OperationTypesRepository.assign_repo('model', 'gpu_models_repository.json')
            repository = OperationTypesRepository().assign_repo('model', 'gpu_models_repository.json')
            available_operations = repository.suitable_operation(task_type=self.task.task_type)

        return available_operations

    def _remove_heavy_operations(self, excluded_models_dict, available_operations) -> list:
        """ Remove operations from operations list """
        if self.preset in excluded_models_dict.keys():
            excluded_operations = excluded_models_dict[self.preset]
            available_operations = [_ for _ in available_operations if _ not in excluded_operations]

        return available_operations

