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

        # Is there a modification in preset or not
        self.modification_using = False

    def composer_params_based_on_preset(self, composer_params: dict) -> dict:
        """ Return composer parameters dictionary with appropriate operations
        based on defined preset
        """
        updated_params = copy(composer_params)

        if self.preset_name is None and 'preset' in updated_params:
            self.preset_name = updated_params['preset']

        if self.preset_name is not None and 'available_operations' not in composer_params:
            available_operations = self._filter_operations_by_preset()
            updated_params['available_operations'] = available_operations

        if updated_params['with_tuning']:
            updated_params['with_tuning'] = True

        return updated_params

    def _filter_operations_by_preset(self):
        """ Filter operations by preset, remove "heavy" operations and save
        appropriate ones
        """
        # TODO remove workaround
        # Use best_quality preset but exclude several operations
        preset_name = self.preset_name
        if 'stable' in self.preset_name:
            # Use best_quality preset but exclude several operations
            preset_name = 'best_quality'
        excluded = ['mlp', 'svc', 'svr', 'arima', 'exog_ts', 'text_clean',
                    'catboost', 'lda', 'qda', 'lgbm', 'one_hot_encoding',
                    'resample', 'stl_arima']

        if '*' in preset_name:
            self.modification_using = True
            # The modification has been added
            preset_name, modification = preset_name.split('*')
            modification = ''.join(('*', modification))

            mod_operations = get_operations_for_task(self.task, mode='all', preset=modification)

        # Get operations
        available_operations = get_operations_for_task(self.task, mode='all', preset=preset_name)

        if self.modification_using:
            # Find subsample of operations
            filtered_operations = set(available_operations).intersection(set(mod_operations))
            available_operations = list(filtered_operations)

        # Exclude "heavy" operations if necessary
        if 'stable' in self.preset_name:
            available_operations = self.new_operations_without_heavy(excluded, available_operations)

        if 'gpu' in self.preset_name:
            repository = OperationTypesRepository().assign_repo('model', 'gpu_models_repository.json')
            available_operations = repository.suitable_operation(task_type=self.task.task_type)

        return available_operations

    @staticmethod
    def new_operations_without_heavy(excluded_operations, available_operations) -> list:
        """ Create new list without heavy operations """
        available_operations = [_ for _ in available_operations if _ not in excluded_operations]

        return available_operations
