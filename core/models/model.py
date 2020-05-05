from abc import ABC
from copy import copy
from dataclasses import dataclass

import numpy as np

from core.models.data import (
    InputData,
)
from core.models.evaluation.evaluation import SkLearnClassificationStrategy, \
    StatsModelsAutoRegressionStrategy, SkLearnRegressionStrategy, SkLearnClusteringStrategy
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.model_types_repository import ModelTypesRepository
from core.repository.task_types import TaskTypesEnum, MachineLearningTasksEnum, \
    compatible_task_types


@dataclass
class Model(ABC):

    def __init__(self, model_type: ModelTypesIdsEnum):
        self.model_type = model_type
        self._eval_strategy, self._data_preprocessing = None, None

    @property
    def description(self):
        model_type = self.model_type
        model_params = 'defaultparams'
        return f'n_{model_type}_{model_params}'

    def _init(self, task: TaskTypesEnum):
        self._eval_strategy = \
            _eval_strategy_for_task(self.model_type, task)

    def fit(self, data: InputData):
        self._init(data.task_type)

        fitted_model = self._eval_strategy.fit(model_type=self.model_type,
                                               train_data=data)
        predict_train = self._eval_strategy.predict(trained_model=fitted_model,
                                                    predict_data=data)
        return fitted_model, predict_train

    def predict(self, fitted_model, data: InputData):
        self._init(data.task_type)

        prediction = self._eval_strategy.predict(trained_model=fitted_model,
                                                 predict_data=data)

        if any([np.isnan(_) for _ in prediction]):
            print("Value error")

        return prediction

    def __str__(self):
        return f'{self.model_type.name}'


def _eval_strategy_for_task(model_type: ModelTypesIdsEnum, task_type_for_data: TaskTypesEnum):
    strategies_for_tasks = {
        MachineLearningTasksEnum.classification: SkLearnClassificationStrategy,
        MachineLearningTasksEnum.regression: SkLearnRegressionStrategy,
        MachineLearningTasksEnum.auto_regression: StatsModelsAutoRegressionStrategy,
        MachineLearningTasksEnum.clustering: SkLearnClusteringStrategy
    }

    models_repo = ModelTypesRepository()
    _, model_info = models_repo.search_models(
        desired_ids=[model_type])

    task_type_for_model = task_type_for_data
    task_types_acceptable_for_model = model_info[0].task_type

    # if the model can't be used directly for the task type from data
    if task_type_for_model not in task_types_acceptable_for_model:
        # search the supplementary task types, that can be included in chain which solves original task
        globally_compatible_task_types = compatible_task_types(task_type_for_model)
        compatible_task_types_acceptable_for_model = list(set(task_types_acceptable_for_model).intersection
                                                          (set(globally_compatible_task_types)))
        if len(compatible_task_types_acceptable_for_model) == 0:
            raise ValueError(f'Model {model_type} can not be used as a part of {task_type_for_model}.')
        task_type_for_model = compatible_task_types_acceptable_for_model[0]

    eval_strategy = strategies_for_tasks[task_type_for_model](model_type)

    return eval_strategy
