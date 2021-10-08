from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.tasks import Task


class ApiPresetHelper:

    def filter_operations_by_preset(self,
                                    task,
                                    preset: str):

        """ Function filter operations by preset, remove "heavy" operations and save
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
        available_operations = get_operations_for_task(task, mode='all')
        available_data_operation = get_operations_for_task(task, mode='data_operation')

        # Exclude "heavy" operations if necessary
        if preset in excluded_models_dict.keys():
            excluded_operations = excluded_models_dict[preset]
            available_operations = [_ for _ in available_operations if _ not in excluded_operations]

        # Save only "light" operations
        if preset in ['ultra_light', 'ultra_light_tun', 'ultra_steady_state']:
            light_models = ['dt', 'dtreg', 'logit', 'linear', 'lasso', 'ridge', 'knn', 'ar']
            included_operations = light_models + available_data_operation
            available_operations = [_ for _ in available_operations if _ in included_operations]

        if preset == 'gpu':
            # OperationTypesRepository.assign_repo('model', 'gpu_models_repository.json')
            repository = OperationTypesRepository().assign_repo('model', 'gpu_models_repository.json')
            available_operations = repository.suitable_operation(task_type=task.task_type)

        return available_operations

    def get_preset(self,
                   task: Task,
                   preset: list,
                   composer_params: dict):
        if preset is None and 'preset' in composer_params:
            preset = composer_params['preset']

        if 'preset' in composer_params:
            del composer_params['preset']

        if preset is not None:
            available_operations = self.filter_operations_by_preset(task, preset)
            composer_params['available_operations'] = available_operations

        if composer_params['with_tuning'] or '_tun' in preset:
            composer_params['with_tuning'] = True

        return composer_params
