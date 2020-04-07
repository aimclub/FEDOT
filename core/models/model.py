from abc import ABC
from copy import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.models.data import (
    InputData,
)
from core.models.evaluation import EvaluationStrategy, SkLearnEvaluationStrategy
from core.models.preprocessing import scaling_preprocess
from core.repository.dataset_types import (
    DataTypesEnum, NumericalDataTypesEnum
)
from core.repository.model_types_repository import ModelTypesIdsEnum


@dataclass
class Model(ABC):
    model_type: ModelTypesIdsEnum
    input_type: DataTypesEnum
    output_type: DataTypesEnum
    eval_strategy: EvaluationStrategy = None
    data_preprocessing: Callable = scaling_preprocess

    def fit(self, data: InputData):
        preprocessed_data = copy(data)
        preprocessed_data.features = self.data_preprocessing(preprocessed_data.features)
        fitted_model = self.eval_strategy.fit(model_type=self.model_type,
                                              train_data=preprocessed_data)
        predict_train = self.eval_strategy.predict(trained_model=fitted_model,
                                                   predict_data=preprocessed_data)
        return fitted_model, predict_train

    def predict(self, fitted_model, data: InputData):
        preprocessed_data = copy(data)
        preprocessed_data.features = self.data_preprocessing(preprocessed_data.features)

        prediction = self.eval_strategy.predict(trained_model=fitted_model,
                                                predict_data=preprocessed_data)

        if any([np.isnan(_) for _ in prediction]):
            print("Value error")

        return prediction

    def __str__(self):
        return f'{self.model_type.name}'


def sklearn_model_by_type(model_type: ModelTypesIdsEnum):
    return Model(model_type=model_type,
                 input_type=NumericalDataTypesEnum.table,
                 output_type=NumericalDataTypesEnum.vector,
                 eval_strategy=SkLearnEvaluationStrategy())
