from enum import Enum
from typing import List


class TaskTypesEnum(Enum):
    pass


class MachineLearningTasksEnum(TaskTypesEnum):
    classification = 'classification',
    regression = 'regression',
    auto_regression = 'auto_regression',
    clustering = 'clustering'  # not applicable as main task yet


class SystemDynamicTasksEnum(TaskTypesEnum):
    models_indent = 'models_identification'
    links_indent = 'links_indention'


# local tasks that can be solved as a part of global tasks
def compatible_task_types(main_task_type: TaskTypesEnum) -> List[TaskTypesEnum]:
    _compatible_task_types = {
        MachineLearningTasksEnum.auto_regression: [MachineLearningTasksEnum.regression],
        MachineLearningTasksEnum.classification: [MachineLearningTasksEnum.clustering],
        MachineLearningTasksEnum.regression: [MachineLearningTasksEnum.clustering]
    }
    if main_task_type not in _compatible_task_types:
        return []
    return _compatible_task_types[main_task_type]
