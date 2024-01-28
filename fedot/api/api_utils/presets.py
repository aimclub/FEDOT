from copy import copy
from enum import Enum
from typing import Optional, List, Union

from fedot.api.time import ApiTime
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_tags_n_repo_enums import ComplexityTags, PresetsTagsEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.tasks import Task


class PresetsEnum(Enum):
    # TODO add test to check that PresetsEnum is accordance with PresetsTagsEnum
    # TODO add ability to reduce preset depends on remaining time
    # TODO add some presets for models with different speed
    AUTO = 'auto', 0
    BEST_QUALITY = 'best_quality', 1
    FAST_TRAIN = 'fast_train', 2
    TREE = '*tree', 3  # workaround for old tree preset
    GPU = 'gpu', None


class OperationsPreset:
    """ Class for presets processing. Preset is a set of operations (data operations
    and models), which will be used during pipeline structure search
    """

    def __init__(self, task: Task, preset_name: Union[List[PresetsEnum], PresetsEnum]):
        self.task = task
        self.preset_name = preset_name

        # Is there a modification in preset or not
        self.modification_using = False
    
    @property
    def preset_name(self):
        return self._preset_name
    
    @preset_name.setter
    def preset_name(self, value):
        if not isinstance(value, PresetsEnum):
            raise ValueError(f"`preset_name` should be `PresetsEnum`, get {type(value)} instead")
        
        if value is PresetsEnum.AUTO:
            value = PresetsEnum.FAST_TRAIN

        self._preset_name = value

    def composer_params_based_on_preset(self, api_params: dict, data_type: Optional[DataTypesEnum] = None) -> dict:
        """ Return composer parameters dictionary with appropriate operations
        based on defined preset
        """
        updated_params = copy(api_params)
        self.preset_name = updated_params.get('preset', self.preset_name)

        if 'available_operations' not in updated_params:
            available_operations = self.filter_operations_by_preset(data_type)
            updated_params['available_operations'] = available_operations

        return updated_params

    def filter_operations_by_preset(self, data_type: Optional[DataTypesEnum] = None):
        """ Filter operations by preset, remove "heavy" operations and save
        appropriate ones
        """
        preset_name = self.preset_name
        operation_repo = None
        tags = list()
        forbidden_tags = list()

        if preset_name is PresetsEnum.GPU:
            # TODO define how GPU preset should works
            operation_repo = OperationTypesRepository.DEFAULT_GPU

        if preset_name is PresetsEnum.FAST_TRAIN:
            forbidden_tags.extend([ComplexityTags.unstable, ComplexityTags.expensive])

        # add tag with preset name
        if preset_name is not PresetsEnum.BEST_QUALITY:
            tags.append(_get_preset_tag(preset_name))

        # Get operations
        available_operations = get_operations_for_task(task=self.task,
                                                       data_type=data_type,
                                                       operation_repo=operation_repo,
                                                       tags=tags or None,
                                                       forbidden_tags=forbidden_tags or None)
        return sorted(available_operations)


def change_preset_based_on_initial_fit(timer: ApiTime, n_jobs: int) -> str:
    """
    If preset was set as 'auto', based on initial pipeline fit time, appropriate one can be chosen
    """
    if timer.time_for_automl in [-1, None]:
        return PresetsEnum.BEST_QUALITY

    # Change preset to appropriate one

    if timer.have_time_for_the_best_quality(n_jobs=n_jobs):
        # It is possible to train only few number of pipelines during optimization - use simplified preset
        return PresetsEnum.BEST_QUALITY
    else:
        return PresetsEnum.FAST_TRAIN
    

def _get_preset_tag(preset: PresetsEnum):
    # never falls because there is the test that checks accordance between PresetsTagsEnum and PresetsEnum
    return next(tag for tag in PresetsTagsEnum if tag.name == preset.value[0])
