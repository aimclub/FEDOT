from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression as SklearnLogReg
import numpy as np
from core.datastream import DataStream


class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, data):
        pass


class LogRegression(EvaluationStrategy):
    def __init__(self, seed=None):
        self.model = SklearnLogReg(random_state=seed)
        self.model_fit = None
        self.model_predict = None

    def evaluate(self, data) -> np.array:
        train_data_x = split_train_data(data.x)
        train_data_y = split_train_data(data.y)
        test_data_x = split_test_data(data.x)
        self.fit(self.model, train_data_x, train_data_y)
        return self.predict(self.model, test_data_x)

    def fit(self, model: SklearnLogReg, fit_data_x: np.array, fit_data_y: np.array):
        self.model_fit = model.fit(fit_data_x, fit_data_y)
        print('Model fit: ', self.model_fit)

    def predict(self, model: SklearnLogReg, data: np.array):
        print('Model prediction')
        self.model_predict = model.predict(data)
        return self.model_predict


class LinRegression(EvaluationStrategy):
    def __init__(self):
        pass

    def evaluate(self, data):
        return self.predict()

    def fit(self):
        pass

    def predict(self):
        return 'LinRegPredict'


class XGBoost(EvaluationStrategy):
    def __init__(self):
        pass

    def evaluate(self, data):
        return self.predict()

    def fit(self):
        pass

    def predict(self):
        return 'XGBoostPredict'


def split_train_data(data, split_ratio=0.8):
    split_point = int(len(data) * split_ratio)
    return data[:split_point]


def split_test_data(data, split_ratio=0.8):
    split_point = int(len(data) * split_ratio)
    return data[split_point:]
