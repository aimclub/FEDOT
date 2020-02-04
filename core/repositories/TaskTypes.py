from enum import Enum


class TaskTypesEnum(Enum):
    pass


class MachineLearningTasks(TaskTypesEnum):
    classification = "classification"
    regression = "regression"


class SystemDynamicTasks(TaskTypesEnum):
    models_indent = "models_indentification"
    links_indent = "links_indention"
