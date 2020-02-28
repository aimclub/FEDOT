from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from core.models.data import (InputData, normalize, split_train_test)
from core.repository.dataset_types import (
    DataTypesEnum, NumericalDataTypesEnum
)


@dataclass
class Model(ABC):
    input_type: DataTypesEnum
    output_type: DataTypesEnum
    __model = None

    @abstractmethod
    def predict(self, data):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, data):
        raise NotImplementedError()

    @abstractmethod
    def tune(self, data):
        raise NotImplementedError()

    def __str__(self):
        return f'{self.__class__.__name__}'


class LogRegression(Model):
    def __init__(self):
        input_type = NumericalDataTypesEnum.table
        output_type = NumericalDataTypesEnum.vector

        super().__init__(input_type=input_type, output_type=output_type)
        self.__model = SklearnLogReg(random_state=1, solver='liblinear', max_iter=100,
                                     tol=1e-3, verbose=1)

    def predict(self, data: InputData):
        predicted = self.__model.predict(data.features)
        return predicted

    def fit(self, data: InputData):
        train_data, _ = train_test_data_setup(data=data)
        self.__model.fit(train_data.features, train_data.target)

    def tune(self, data):
        return 1


class XGBoost(Model):
    def __init__(self):
        input_type = NumericalDataTypesEnum.table
        output_type = NumericalDataTypesEnum.vector
        super().__init__(input_type=input_type, output_type=output_type)

    def predict(self, data):
        pass

    def fit(self, data):
        pass

    def tune(self, data):
        pass


class RandomForest(Model):
    def __init__(self):
        input_type = NumericalDataTypesEnum.table
        output_type = NumericalDataTypesEnum.vector

        super().__init__(input_type=input_type, output_type=output_type)
        self.__model = RandomForestClassifier(n_estimators=100, max_depth=2, n_jobs=-1)

    def predict(self, data: InputData):
        predicted = self.__model.predict(data.features)
        return predicted

    def fit(self, data: InputData):
        train_data, _ = train_test_data_setup(data=data)
        self.__model.fit(train_data.features, train_data.target)

    def tune(self, data):
        return 1


class DecisionTree(Model):
    def __init__(self):
        input_type = NumericalDataTypesEnum.table
        output_type = NumericalDataTypesEnum.vector

        super().__init__(input_type=input_type, output_type=output_type)
        self.__model = DecisionTreeClassifier(max_depth=2, )

    def predict(self, data: InputData):
        predicted = self.__model.predict(data.features)
        return predicted

    def fit(self, data: InputData):
        train_data, _ = train_test_data_setup(data=data)
        self.__model.fit(train_data.features, train_data.target)

    def tune(self, data):
        return 1


class LDA(Model):
    def __init__(self):
        input_type = NumericalDataTypesEnum.table
        output_type = NumericalDataTypesEnum.vector

        super().__init__(input_type=input_type, output_type=output_type)
        self.__model = LinearDiscriminantAnalysis(solver="svd")

    def predict(self, data: InputData):
        predicted = self.__model.predict(data.features)
        return predicted

    def fit(self, data: InputData):
        train_data, _ = train_test_data_setup(data=data)
        self.__model.fit(train_data.features, train_data.target)

    def tune(self, data):
        return 1


class QDA(Model):
    def __init__(self):
        input_type = NumericalDataTypesEnum.table
        output_type = NumericalDataTypesEnum.vector

        super().__init__(input_type=input_type, output_type=output_type)
        self.__model = QuadraticDiscriminantAnalysis()

    def predict(self, data: InputData):
        predicted = self.__model.predict(data.features)
        return predicted

    def fit(self, data: InputData):
        train_data, _ = train_test_data_setup(data=data)
        self.__model.fit(train_data.features, train_data.target)

    def tune(self, data):
        return 1


class MLP_Classifier():
    def __init__(self):
        input_type = NumericalDataTypesEnum.table
        output_type = NumericalDataTypesEnum.vector

        super().__init__(input_type=input_type, output_type=output_type)
        self.__model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)

    def predict(self, data: InputData):
        predicted = self.__model.predict(data.features)
        return predicted

    def fit(self, data: InputData):
        train_data, _ = train_test_data_setup(data=data)
        self.__model.fit(train_data.features, train_data.target)


def train_test_data_setup(data: InputData) -> Tuple[InputData, InputData]:
    train_data_x, test_data_x = split_train_test(data.features)
    train_data_y, test_data_y = split_train_test(data.target)
    train_idx, test_idx = split_train_test(data.idx)
    train_data = InputData(features=normalize(train_data_x), target=train_data_y,
                           idx=train_idx)
    test_data = InputData(features=normalize(test_data_x), target=test_data_y,
                          idx=test_idx)
    return train_data, test_data
