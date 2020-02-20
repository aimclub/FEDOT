from abc import ABC, abstractmethod

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

from core.models.data import (
    Data,
    split_train_test,
    normalize
)
from core.repository.dataset_types import (
    DataTypesEnum,
    NumericalDataTypesEnum
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


class LogRegression(Model):
    def __init__(self):
        input_type = NumericalDataTypesEnum.table
        output_type = NumericalDataTypesEnum.vector

        super().__init__(input_type=input_type, output_type=output_type)
        self.__model = SklearnLogReg(random_state=1, solver='liblinear',
                                     max_iter=100, tol=1e-3, verbose=1)

    def predict(self, data: Data):
        predicted = self.__model.predict(data.features)
        return predicted

    def fit(self, data: Data):
        features, target, _ = train_test_data_setup(data=data)
        self.__model.fit(features, target)

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


class KNN(Model):
    def __init__(self):
        input_type = NumericalDataTypesEnum.table
        output_type = NumericalDataTypesEnum.vector

        super().__init__(input_type=input_type, output_type=output_type)
        self.__model = SklearnKNN(n_neighbors=15)

    def predict(self, data: Data):
        predicted = self.__model.predict(data.features)
        return predicted

    def fit(self, data: Data):
        features, target, _ = train_test_data_setup(data=data)
        self.__model.fit(features, target)

    def tune(self, data):
        return 1


# TODO: Should return Data-objects
def train_test_data_setup(data: Data):
    train_features, test_features = split_train_test(data.features)
    train_target, _ = split_train_test(data.target)
    return normalize(train_features), train_target, normalize(test_features)
