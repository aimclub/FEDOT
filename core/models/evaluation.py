import warnings

import numpy as np
from sklearn.cluster import KMeans as SklearnKmeans
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso as SklearnLassoReg
from sklearn.linear_model import LinearRegression as SklearnLinReg
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.linear_model import Ridge as SklearnRidgeReg
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from xgboost import XGBClassifier

from core.models.data import InputData, OutputData
from core.repository.model_types_repository import ModelTypesIdsEnum

warnings.filterwarnings("ignore", category=UserWarning)


class EvaluationStrategy:
    def fit(self, model_type: ModelTypesIdsEnum, train_data: InputData):
        raise NotImplementedError()

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        raise NotImplementedError()

    def tune(self, model, data_for_tune: InputData):
        raise NotImplementedError()


class SkLearnEvaluationStrategy(EvaluationStrategy):
    def __init__(self, model_type: ModelTypesIdsEnum):
        self.__model_by_types = {
            ModelTypesIdsEnum.xgboost: XGBClassifier,
            ModelTypesIdsEnum.logit: SklearnLogReg,
            ModelTypesIdsEnum.knn: SklearnKNN,
            ModelTypesIdsEnum.dt: DecisionTreeClassifier,
            ModelTypesIdsEnum.rf: RandomForestClassifier,
            ModelTypesIdsEnum.mlp: MLPClassifier,
            ModelTypesIdsEnum.lda: LinearDiscriminantAnalysis,
            ModelTypesIdsEnum.qda: QuadraticDiscriminantAnalysis,
            ModelTypesIdsEnum.linear: SklearnLinReg,
            ModelTypesIdsEnum.ridge: SklearnRidgeReg,
            ModelTypesIdsEnum.lasso: SklearnLassoReg,
            ModelTypesIdsEnum.kmeans: SklearnKmeans
        }

        self._sklearn_model_impl = self._convert_to_sklearn(model_type)

    def fit(self, model_type: ModelTypesIdsEnum, train_data: InputData):
        sklearn_model = self._sklearn_model_impl()
        sklearn_model.fit(train_data.features, train_data.target.ravel())
        return sklearn_model

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        raise NotImplementedError()

    def tune(self, model, data_for_tune: InputData):
        return model

    def _convert_to_sklearn(self, model_type: ModelTypesIdsEnum):
        if model_type in self.__model_by_types.keys():
            return self.__model_by_types[model_type]
        else:
            raise ValueError(f'Impossible to obtain SKlearn strategy for {model_type}')


class SkLearnClassificationStrategy(SkLearnEvaluationStrategy):
    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        prediction = trained_model.predict_proba(predict_data.features)[:, 1]
        return prediction


class SkLearnRegressionStrategy(SkLearnEvaluationStrategy):
    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        prediction = trained_model.predict(predict_data.features)
        return prediction


class SkLearnClusteringStrategy(SkLearnEvaluationStrategy):
    def fit(self, model_type: ModelTypesIdsEnum, train_data: InputData):
        sklearn_model = self._sklearn_model_impl(n_clusters=2)
        sklearn_model = sklearn_model.fit(train_data.features)
        return sklearn_model

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        prediction = trained_model.predict(predict_data.features)
        return prediction


class StatsModelsAutoRegressionStrategy(EvaluationStrategy):
    def __init__(self, model_type: ModelTypesIdsEnum):
        self._model_functions_by_types = {
            ModelTypesIdsEnum.arima: (self._fit_arima, self._predict_arima),
            ModelTypesIdsEnum.ar: (self._fit_ar, self._predict_ar)
        }
        self._model_specific_fit, self._model_specific_predict = self._init_stats_model_functions(model_type)

    def _init_stats_model_functions(self, model_type: ModelTypesIdsEnum):
        if model_type in self._model_functions_by_types.keys():
            return self._model_functions_by_types[model_type]
        else:
            raise ValueError(f'Impossible to obtain Stats strategy for {model_type}')

    def fit(self, model_type: ModelTypesIdsEnum, train_data: InputData):
        stats_model = self._model_specific_fit(train_data)
        return stats_model

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        return self._model_specific_predict(trained_model, predict_data)

    def _fit_ar(self, train_data: InputData):
        return AutoReg(train_data.target, lags=[1, 2, 6, 12, 24],
                       exog=train_data.features).fit()

    def _fit_arima(self, train_data: InputData):
        return ARIMA(train_data.target, order=(2, 0, 0),
                     exog=train_data.features).fit(disp=0)

    def _predict_ar(self, trained_model, predict_data: InputData) -> OutputData:
        start, end = trained_model.nobs, \
                     trained_model.nobs + len(predict_data.target) - 1
        exog, exog_oos = None, predict_data.features

        if trained_model.data.endog is predict_data.target:
            # if train sample used
            start, end = 0, len(predict_data.target)
            exog, exog_oos = predict_data.features, \
                             predict_data.features

        prediction = trained_model.predict(start=start, end=end,
                                           exog=exog, exog_oos=exog_oos)

        diff = len(predict_data.target) - len(prediction)
        if diff != 0:
            prediction = np.append(prediction, prediction[-diff:])

        return prediction

    def _predict_arima(self, trained_model, predict_data: InputData) -> OutputData:
        start, end = trained_model.nobs, \
                     trained_model.nobs + len(predict_data.target) - 1
        exog = predict_data.features

        if trained_model.data.endog is predict_data.target:
            # if train sample used
            start, end = 0, len(predict_data.target)

        prediction = trained_model.predict(start=start, end=end,
                                           exog=exog)

        return prediction[0:len(predict_data.target)]

    def tune(self, model, data_for_tune: InputData):
        return model
