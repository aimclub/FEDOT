from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression as SklearnLogReg
import numpy as np
from core.datastream import DataStream


class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, data: DataStream):
        pass


class LogRegression(EvaluationStrategy):
    def __init__(self, seed=None):
        self.model = SklearnLogReg(random_state=seed, solver='sag', max_iter=10000)
        self.model_fit = None
        self.model_predict = None

    def evaluate(self, data: DataStream) -> np.array:
        train_data_x, train_data_y, test_data_x = test_train_data_setup(data)
        self.fit(train_data_x, train_data_y)
        return self.predict(test_data_x)

    def fit(self, fit_data_x: np.array, fit_data_y: np.array):
        self.model_fit = self.model.fit(fit_data_x, fit_data_y)
        print('Model fit: ', self.model_fit)

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


def test_train_data_setup(data: DataStream):
    train_data_x, test_data_x = splitted_train_test(data.x)
    train_data_y, _ = splitted_train_test(data.y)
    return train_data_x, train_data_y, test_data_x


def splitted_train_test(data, split_ratio=0.8):
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]
