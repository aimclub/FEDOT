from abc import ABC
from copy import copy
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from core.models.data import (
    InputData,
)
from core.models.evaluation import EvaluationStrategy, SkLearnClassificationStrategy, \
    StatsModelsAutoRegressionStrategy, SkLearnRegressionStrategy, SkLearnClusteringStrategy
from core.models.preprocessing import scaling_preprocess, simple_preprocess
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.model_types_repository import ModelTypesRepository
from core.repository.task_types import TaskTypesEnum, MachineLearningTasksEnum, \
    compatible_task_types


# noinspection SpellCheckingInspection
@dataclass
class Model(ABC):
    model_type: ModelTypesIdsEnum
    eval_strategy: Optional[EvaluationStrategy] = None
    data_preprocessing: Optional[Callable] = None

    def fit(self, data: InputData):
        self.eval_strategy, self.data_preprocessing = \
            _eval_strategy_for_task(self.model_type, data.task_type)
        preprocessed_data = copy(data)
        preprocessed_data.features = self._data_preprocessing(preprocessed_data.features)
        fitted_model = self._eval_strategy.fit(model_type=self.model_type,
                                               train_data=preprocessed_data)
        predict_train = self._eval_strategy.predict(trained_model=fitted_model,
                                                    predict_data=preprocessed_data)
        return fitted_model, predict_train

    def predict(self, fitted_model, data: InputData):
        preprocessed_data = copy(data)
        if not (self._data_preprocessing and self._eval_strategy):
            raise ValueError(f'Model {str(self)} not initialised')
        preprocessed_data.features = self._data_preprocessing(preprocessed_data.features)

        prediction = self._eval_strategy.predict(trained_model=fitted_model,
                                                 predict_data=preprocessed_data)

        if any([np.isnan(_) for _ in prediction]):
            print("Value error")

        return prediction

    def __str__(self):
        return f'{self.model_type.name}'


def init_model_by_type(model_type: ModelTypesIdsEnum):
    return Model(model_type=model_type,
                 eval_strategy=None)


def _eval_strategy_for_task(model_type: ModelTypesIdsEnum, task_type: TaskTypesEnum):
    preprocessing_for_tasks = {
        MachineLearningTasksEnum.auto_regression: simple_preprocess,
        MachineLearningTasksEnum.classification: scaling_preprocess,
        MachineLearningTasksEnum.regression: scaling_preprocess,
        MachineLearningTasksEnum.clustering: scaling_preprocess
    }

    strategies_for_tasks = {
        MachineLearningTasksEnum.classification: SkLearnClassificationStrategy,
        MachineLearningTasksEnum.regression: SkLearnRegressionStrategy,
        MachineLearningTasksEnum.auto_regression: StatsModelsAutoRegressionStrategy,
        MachineLearningTasksEnum.clustering: SkLearnClusteringStrategy
    }

    preprocessing_function = preprocessing_for_tasks.get(task_type, scaling_preprocess)

    models_repo = ModelTypesRepository()
    _, model_info = models_repo.search_models(
        desired_ids=[model_type])

    local_task_types = model_info[0].task_type

    # if the model can't be used for specified task type
    if task_type not in local_task_types:
        globally_compatible_task_types = compatible_task_types(task_type)
        available_task_types = list(set(local_task_types).intersection
                                    (set(globally_compatible_task_types)))
        if not available_task_types:
            raise ValueError(f'Model {model_type} can not be used as a part of {task_type}.')
        task_type = available_task_types[0]

    eval_strategy = strategies_for_tasks[task_type]()

    return eval_strategy, preprocessing_function
