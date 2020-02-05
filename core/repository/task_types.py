from enum import Enum


class TaskTypesEnum(Enum):
    pass


class MachineLearningTasksEnum(TaskTypesEnum):
    classification = 'classification'
    regression = 'regression'


class SystemDynamicTasksEnum(TaskTypesEnum):
    models_indent = 'models_indentification'
    links_indent = 'links_indention'
