import warnings
from typing import Optional

from sklearn.cluster import KMeans as SklearnKmeans
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    RandomForestRegressor
from sklearn.linear_model import Lasso as SklearnLassoReg, LinearRegression as SklearnLinReg, \
    LogisticRegression as SklearnLogReg, Ridge as SklearnRidgeReg, SGDRegressor as SklearnSGD
from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN, KNeighborsRegressor as SklearnKNNReg
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC as SklearnSVC, LinearSVR as SklearnSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from benchmark.benchmark_model_types import BenchmarkModelTypesEnum
from benchmark.tpot.b_tpot import fit_tpot, predict_tpot_class, predict_tpot_reg
from core.models.data import InputData, OutputData
from core.models.evaluation.automl_eval import fit_h2o, predict_h2o
from core.models.evaluation.hyperparams import params_range_by_model
from core.models.evaluation.stats_models_eval import fit_ar, fit_arima, predict_ar, predict_arima
from core.models.tuners import ForecastingCustomRandomTuner, SklearnCustomRandomTuner, SklearnTuner
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.utils import labels_to_dummy_probs

warnings.filterwarnings("ignore", category=UserWarning)


class EvaluationStrategy:

    def fit(self, train_data: InputData):
        raise NotImplementedError()

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        raise NotImplementedError()

    def fit_tuned(self, train_data: InputData):
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

    __metric_by_type = {
        'classification': make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True),
        'regression': make_scorer(mean_squared_error, greater_is_better=False),
    }

    def __init__(self, model_type: ModelTypesIdsEnum):
        self._sklearn_model_impl = self._convert_to_sklearn(model_type)
        self._tune_func: SklearnTuner = Optional[SklearnTuner]
        self.params_for_fit = None
        self.model_type = model_type

    def fit(self, train_data: InputData):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sklearn_model = self._sklearn_model_impl()
        sklearn_model.fit(train_data.features, train_data.target.ravel())
        return sklearn_model

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        raise NotImplementedError()

    def fit_tuned(self, train_data: InputData,
                  iterations: int = 30):
        trained_model = self.fit(train_data=train_data)
        params_range = params_range_by_model.get(self.model_type, None)
        metric = self.__metric_by_type.get(train_data.task_type.name, None)
        self._tune_func = SklearnCustomRandomTuner
        if not params_range:
            self.params_for_fit = None
            return trained_model, trained_model.get_params()

        tuned_params, best_model = self._tune_func().tune(trained_model=trained_model,
                                                          tune_data=train_data,
                                                          params_range=params_range,
                                                          iterations=iterations,
                                                          cross_val_fold_num=5,
                                                          scorer=metric)

        if best_model and tuned_params:
            trained_model = best_model
            self.params_for_fit = tuned_params
        return trained_model, tuned_params

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

        if n_classes < 2:
            raise NotImplementedError()
        elif n_classes == 2:
            prediction = prediction[:, 1]

        return prediction


class SkLearnRegressionStrategy(SkLearnEvaluationStrategy):
    def predict(self, trained_model, predict_data: InputData):
        prediction = trained_model.predict(predict_data.features)
        return prediction


class SkLearnClusteringStrategy(SkLearnEvaluationStrategy):
    def fit(self, train_data: InputData):
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

    __default_params_by_model = {
        ModelTypesIdsEnum.arima: {'order': (2, 0, 0)},
        ModelTypesIdsEnum.ar: {'lags': (1, 2, 6, 12, 24)}
    }
    __params_range_by_model = {
        ModelTypesIdsEnum.arima: {'order': ((2, 0, 0), (5, 0, 5))},
        ModelTypesIdsEnum.ar: {'lags': (range(1, 6), range(6, 96, 6))}
    }

    def __init__(self, model_type: ModelTypesIdsEnum):
        self._model_specific_fit, self._model_specific_predict = self._init_stats_model_functions(model_type)
        self._params_range = self.__params_range_by_model[model_type]
        self._default_params = self.__default_params_by_model[model_type]

        self.params_for_fit = None

    def _init_stats_model_functions(self, model_type: ModelTypesIdsEnum):
        if model_type in self._model_functions_by_types.keys():
            return self._model_functions_by_types[model_type]
        else:
            raise ValueError(f'Impossible to obtain Stats strategy for {model_type}')

    def fit(self, train_data: InputData):
        stats_model = self._model_specific_fit(train_data, self._default_params)
        self.params_for_fit = self._default_params
        return stats_model

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        return self._model_specific_predict(trained_model, predict_data)

    def fit_tuned(self, train_data: InputData, iterations: int = 10):
        tuned_params = ForecastingCustomRandomTuner().tune(fit=self._model_specific_fit,
                                                           predict=self._model_specific_predict,
                                                           tune_data=train_data,
                                                           params_range=self._params_range,
                                                           default_params=self._default_params,
                                                           iterations=iterations)

        stats_model = self._model_specific_fit(train_data, tuned_params)
        self.params_for_fit = tuned_params

        return stats_model, tuned_params


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

    def fit(self, train_data: InputData):
        benchmark_model = self._model_specific_fit(train_data)
        return benchmark_model

    def predict(self, trained_model, predict_data: InputData):
        return self._model_specific_predict(trained_model, predict_data)

    def fit_tuned(self, train_data: InputData):
        raise NotImplementedError()


class AutoMLRegressionStrategy(AutoMLEvaluationStrategy):
    _model_functions_by_type = {
        ModelTypesIdsEnum.tpot: (fit_tpot, predict_tpot_reg),
        ModelTypesIdsEnum.h2o: (fit_h2o, predict_h2o)
    }
