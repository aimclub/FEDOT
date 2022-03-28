import datetime

from copy import copy
from typing import Union

from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import \
    PipelineComposerRequirements
from fedot.core.constants import BEST_QUALITY_PRESET_NAME, \
    FAST_TRAIN_PRESET_NAME, MINIMAL_PIPELINE_NUMBER_FOR_EVALUATION
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
            preset_name = BEST_QUALITY_PRESET_NAME
        excluded = ['mlp', 'svc', 'svr', 'arima', 'exog_ts', 'text_clean',
                    'catboost', 'lda', 'qda', 'lgbm', 'one_hot_encoding',
                    'resample', 'stl_arima']
        excluded_tree = ['xgboost', 'catboost', 'xgbreg', 'catboostreg']

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

        filtered_operations = set(available_operations).difference(set(excluded_tree))
        available_operations = list(filtered_operations)

        return available_operations

    @staticmethod
    def new_operations_without_heavy(excluded_operations, available_operations) -> list:
        """ Create new list without heavy operations """
        available_operations = [_ for _ in available_operations if _ not in excluded_operations]

        return available_operations


def update_builder(builder: ComposerBuilder,
                   composer_requirements: PipelineComposerRequirements,
                   fit_time: datetime.timedelta,
                   full_minutes_timeout: Union[int, None], preset: str) -> [ComposerBuilder, str]:
    """ Updates the builder if a preset needs to be set automatically """
    if preset != 'auto':
        return builder, preset

    # Find appropriate preset
    new_preset = change_preset_based_on_initial_fit(fit_time, full_minutes_timeout)

    preset_manager = OperationsPreset(task=builder.task, preset_name=new_preset)
    new_operations = preset_manager.composer_params_based_on_preset(composer_params={'preset': new_preset})
    # Insert updated operations list into source composer parameters
    composer_requirements.primary = new_operations['available_operations']
    composer_requirements.secondary = copy(new_operations['available_operations'])
    builder.with_requirements(composer_requirements)
    return builder, new_preset


def change_preset_based_on_initial_fit(fit_time: datetime.timedelta,
                                       full_minutes_timeout: Union[int, None]) -> str:
    """
    If preset was set as 'auto', based on initial pipeline fit time, appropriate one can be chosen

    :param fit_time: spend time for fit initial pipeline
    :param full_minutes_timeout: minutes for AutoML algorithm
    """
    if full_minutes_timeout in [-1, None]:
        return BEST_QUALITY_PRESET_NAME

    # Change preset to appropriate one
    init_fit_minutes = fit_time.total_seconds() / 60
    minimal_minutes_for_all_calculations = init_fit_minutes * MINIMAL_PIPELINE_NUMBER_FOR_EVALUATION
    if minimal_minutes_for_all_calculations > full_minutes_timeout:
        # It is possible to train only few number of pipelines during optimization - use simplified preset
        return FAST_TRAIN_PRESET_NAME
    else:
        return BEST_QUALITY_PRESET_NAME
