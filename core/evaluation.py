from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as SklearnLogReg

from core.data import Data, DataStream


class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, data: DataStream):
        pass


class LogRegression(EvaluationStrategy):
    def __init__(self, seed=None):
        self.model = SklearnLogReg(random_state=seed, solver='liblinear', max_iter=10000)
        self.model_predict = None

    def evaluate(self, data: DataStream) -> np.array:
        train_data, test_data = train_test_data_setup(data)
        self.fit(train_data.features, train_data.target)
        return self.predict(test_data.features)

    def fit(self, fit_data_x: np.array, fit_data_y: np.array):
        self.model.fit(fit_data_x, fit_data_y)
        print('Model fit: ', self.model)

    def predict(self, data: np.array):
        self.model_predict = self.model.predict(data)
        print('Model prediction: ', self.model_predict)
        return self.model_predict


class LinRegression(EvaluationStrategy):
    def __init__(self):
        pass

    def evaluate(self, data: DataStream):
        return self.predict()

    def fit(self):
        pass

    def predict(self):
        return 'LinRegPredict'


class XGBoost(EvaluationStrategy):
    def __init__(self):
        pass

    def evaluate(self, data: DataStream):
        return self.predict()

    def fit(self):
        pass

    def predict(self):
        return 'XGBoostPredict'


def train_test_data_setup(data: DataStream) -> Tuple[Data, Data]:
    train_data_x, test_data_x = split_train_test(data.x)
    train_data_y, test_data_y = split_train_test(data.y)
    train_data = Data(features=normalize(train_data_x), target=train_data_y)
    test_data = Data(features=normalize(test_data_x), target=test_data_y)
    return train_data, test_data


def split_train_test(data: np.array, split_ratio=0.8):
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]


def normalize(x):
    """Normalize data with sklearn.preprocessing.scale()"""
    return preprocessing.scale(x)
