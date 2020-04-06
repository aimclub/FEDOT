from abc import ABC
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from core.models.data import (
    InputData,
    split_train_test,
)
from core.models.evaluation import EvaluationStrategy, SkLearnEvaluationStrategy
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

    def fit(self, data: InputData):
        fitted_model = self.eval_strategy.fit(model_type=self.model_type,
                                              train_data=data)
        predict_train = self.eval_strategy.predict(trained_model=fitted_model,
                                                   predict_data=data)
        return fitted_model, predict_train

    def predict(self, fitted_model, data: InputData):
        prediction = self.eval_strategy.predict(trained_model=fitted_model,
                                                predict_data=data)

        if any([np.isnan(_) for _ in prediction]):
            print("Value error")

        return prediction

    def __str__(self):
        return f'{self.model_type.name}'


def preprocess(x):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(x)
    x = imp.transform(x)
    return preprocessing.scale(x)


def train_test_data_setup(data: InputData) -> Tuple[InputData, InputData]:
    train_data_x, test_data_x = split_train_test(data.features)
    train_data_y, test_data_y = split_train_test(data.target)
    train_idx, test_idx = split_train_test(data.idx)
    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=train_idx)
    test_data = InputData(features=test_data_x, target=test_data_y, idx=test_idx)
    return train_data, test_data


def sklearn_model_by_type(model_type: ModelTypesIdsEnum):
    return Model(model_type=model_type,
                 input_type=NumericalDataTypesEnum.table,
                 output_type=NumericalDataTypesEnum.vector,
                 eval_strategy=SkLearnEvaluationStrategy())
