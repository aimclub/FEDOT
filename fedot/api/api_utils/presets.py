from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task
from fedot.api.api_utils.presets_types import excluded_models_dict, light_models


class API_preset_helper():

    def filter_operations_by_preset(self,
                                    task,
                                    preset: str):
        """ Function filter operations by preset, remove "heavy" operations and save
        appropriate ones
        """

        # Get data operations and models
        available_operations = get_operations_for_task(task, mode='all')
        available_data_operation = get_operations_for_task(task, mode='data_operation')

        # Exclude "heavy" operations if necessary
        if preset in excluded_models_dict.keys():
            excluded_operations = excluded_models_dict[preset]
            available_operations = [_ for _ in available_operations if _ not in excluded_operations]

            # Save only "light" operations
        if preset in ['ultra_light', 'ultra_light_tun']:
            included_operations = light_models + available_data_operation
            available_operations = [_ for _ in available_operations if _ in included_operations]

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
            composer_params['with_tuning'] = '_tun' in preset or preset is None

        return composer_params
