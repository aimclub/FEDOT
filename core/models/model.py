from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from core.models.data import (
    InputData,
    split_train_test,
)
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


class SkLearnModel(Model):

    def __init__(self):
        input_type = NumericalDataTypesEnum.table
        output_type = NumericalDataTypesEnum.vector
        super().__init__(input_type=input_type, output_type=output_type)

        self.__model = self.initialized_model()

    @abstractmethod
    def initialized_model(self):
        raise NotImplementedError()

    def predict(self, data: InputData):
        prediction = self.__model.predict_proba(data.features)
        return prediction[:, 1] if prediction.shape[1] > 1 else prediction

    def fit(self, data: InputData):
        train_data, _ = train_test_data_setup(data=data)
        self.__model.fit(train_data.features, train_data.target)

    def tune(self, data):
        return 1


class LogRegression(SkLearnModel):
    def initialized_model(self):
        return SklearnLogReg(random_state=1, solver='liblinear', max_iter=100,
                             tol=1e-3, verbose=0)


class XGBoost(SkLearnModel):
    def initialized_model(self):
        return XGBClassifier()


class RandomForest(SkLearnModel):
    def initialized_model(self):
        return RandomForestClassifier(n_estimators=100, max_depth=2, n_jobs=-1)


class DecisionTree(SkLearnModel):
    def initialized_model(self):
        return DecisionTreeClassifier(max_depth=2, )


class KNN(SkLearnModel):
    def initialized_model(self):
        return SklearnKNN(n_neighbors=15)


class LDA(SkLearnModel):
    def initialized_model(self):
        return LinearDiscriminantAnalysis(solver="svd")


class QDA(SkLearnModel):
    # TODO investigate NaN in results
    def initialized_model(self):
        return QuadraticDiscriminantAnalysis()


class MLP(SkLearnModel):
    def initialized_model(self):
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)


def train_test_data_setup(data: InputData) -> Tuple[InputData, InputData]:
    train_data_x, test_data_x = split_train_test(data.features)
    train_data_y, test_data_y = split_train_test(data.target)
    train_idx, test_idx = split_train_test(data.idx)
    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=train_idx)
    test_data = InputData(features=test_data_x, target=test_data_y, idx=test_idx)
    return train_data, test_data
