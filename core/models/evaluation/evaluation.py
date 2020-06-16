import warnings
import pandas as pd

from benchmark.benchmark_model_types import BenchmarkModelTypesEnum
from benchmark.tpot.b_tpot import fit_tpot, predict_tpot_reg, predict_tpot_class
from core.models.data import InputData, OutputData
from core.models.evaluation.automl_eval import fit_h2o, predict_h2o
from core.models.evaluation.stats_models_eval import fit_ar, fit_arima, predict_ar, predict_arima
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.utils import labels_to_dummy_probs
from sklearn.cluster import KMeans as SklearnKmeans
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso as SklearnLassoReg
from sklearn.linear_model import LinearRegression as SklearnLinReg
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.linear_model import Ridge as SklearnRidgeReg
from sklearn.linear_model import SGDRegressor as SklearnSGD
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.neighbors import KNeighborsRegressor as SklearnKNNReg
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC as SklearnSVC
from sklearn.svm import LinearSVR as SklearnSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)


class EvaluationStrategy:
    def fit(self, model_type: ModelTypesIdsEnum, train_data: InputData):
        raise NotImplementedError()

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        raise NotImplementedError()

    def tune(self, model, data_for_tune: InputData):
        raise NotImplementedError()


class SkLearnEvaluationStrategy(EvaluationStrategy):
    __model_by_types = {
        ModelTypesIdsEnum.xgboost: XGBClassifier,
        ModelTypesIdsEnum.xgbreg: XGBRegressor,
        ModelTypesIdsEnum.adareg: AdaBoostRegressor,
        ModelTypesIdsEnum.gbr: GradientBoostingRegressor,
        ModelTypesIdsEnum.logit: SklearnLogReg,
        ModelTypesIdsEnum.knn: SklearnKNN,
        ModelTypesIdsEnum.knnreg: SklearnKNNReg,
        ModelTypesIdsEnum.dt: DecisionTreeClassifier,
        ModelTypesIdsEnum.dtreg: DecisionTreeRegressor,
        ModelTypesIdsEnum.treg: ExtraTreesRegressor,
        ModelTypesIdsEnum.rf: RandomForestClassifier,
        ModelTypesIdsEnum.rfreg: RandomForestRegressor,
        ModelTypesIdsEnum.mlp: MLPClassifier,
        ModelTypesIdsEnum.lda: LinearDiscriminantAnalysis,
        ModelTypesIdsEnum.qda: QuadraticDiscriminantAnalysis,
        ModelTypesIdsEnum.linear: SklearnLinReg,
        ModelTypesIdsEnum.ridge: SklearnRidgeReg,
        ModelTypesIdsEnum.lasso: SklearnLassoReg,
        ModelTypesIdsEnum.kmeans: SklearnKmeans,
        ModelTypesIdsEnum.svc: SklearnSVC,
        ModelTypesIdsEnum.svr: SklearnSVR,
        ModelTypesIdsEnum.sgdr: SklearnSGD,

    }

    def __init__(self, model_type: ModelTypesIdsEnum):
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
    __models_without_prob = [SklearnSVC]

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        n_classes = len(trained_model.classes_)
        if type(trained_model) in self.__models_without_prob:
            prediction = trained_model.predict(predict_data.features)
            prediction = labels_to_dummy_probs(prediction)
        else:
            prediction = trained_model.predict_proba(predict_data.features)

        if n_classes <= 2:
            prediction = prediction[:, 1]

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
    _model_functions_by_types = {
        ModelTypesIdsEnum.arima: (fit_arima, predict_arima),
        ModelTypesIdsEnum.ar: (fit_ar, predict_ar)
    }

    def __init__(self, model_type: ModelTypesIdsEnum):
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

    def tune(self, model, data_for_tune: InputData):
        return model


class AutoMLEvaluationStrategy(EvaluationStrategy):
    _model_functions_by_type = {
        ModelTypesIdsEnum.tpot: (fit_tpot, predict_tpot_class),
        ModelTypesIdsEnum.h2o: (fit_h2o, predict_h2o)
    }

    def __init__(self, model_type: BenchmarkModelTypesEnum):
        self._model_specific_fit, self._model_specific_predict = self._init_benchmark_model_functions(model_type)

    def _init_benchmark_model_functions(self, model_type):
        if model_type in self._model_functions_by_type.keys():
            return self._model_functions_by_type[model_type]
        else:
            raise ValueError(f'Impossible to obtain benchmark strategy for {model_type}')

    def fit(self, model_type: BenchmarkModelTypesEnum, train_data: InputData):
        benchmark_model = self._model_specific_fit(train_data)
        return benchmark_model

    def predict(self, trained_model, predict_data: InputData):
        return self._model_specific_predict(trained_model, predict_data)

    def tune(self, model, data_for_tune):
        raise NotImplementedError()


class AutoMLRegressionStrategy(AutoMLEvaluationStrategy):
    _model_functions_by_type = {
        ModelTypesIdsEnum.tpot: (fit_tpot, predict_tpot_reg),
        ModelTypesIdsEnum.h2o: (fit_h2o, predict_h2o)
    }
