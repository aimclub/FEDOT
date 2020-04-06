from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from core.models.data import InputData
from core.repository.model_types_repository import ModelTypesIdsEnum


class EvaluationStrategy:
    def __init__(self):
        pass

    def fit(self, model_type: ModelTypesIdsEnum, train_data: InputData):
        raise NotImplementedError()

    def predict(self, trained_model, predict_data: InputData):
        raise NotImplementedError()

    def tune(self, model, data_for_tune: InputData):
        raise NotImplementedError()


class SkLearnEvaluationStrategy(EvaluationStrategy):
    def __init__(self):
        super().__init__()
        self.__model_by_types = {
            ModelTypesIdsEnum.xgboost: XGBClassifier,
            ModelTypesIdsEnum.logit: SklearnLogReg,
            ModelTypesIdsEnum.knn: SklearnKNN,
            ModelTypesIdsEnum.dt: DecisionTreeClassifier,
            ModelTypesIdsEnum.rf: RandomForestClassifier,
            ModelTypesIdsEnum.mlp: MLPClassifier,
            ModelTypesIdsEnum.lda: LinearDiscriminantAnalysis,
            ModelTypesIdsEnum.qda: QuadraticDiscriminantAnalysis
        }

    def fit(self, model_type: ModelTypesIdsEnum, train_data: InputData):
        sklearn_model = self._convert_to_sklearn(model_type)
        sklearn_model.fit(train_data.features, train_data.target.ravel())
        return sklearn_model

    def predict(self, trained_model, predict_data: InputData):
        prediction = trained_model.predict_proba(predict_data.features)[:, 1]
        return prediction

    def tune(self, model, data_for_tune: InputData):
        return model

    def _convert_to_sklearn(self, model_type: ModelTypesIdsEnum):
        if model_type in self.__model_by_types.keys():
            return self.__model_by_types[model_type]()
        else:
            raise ValueError()
