import warnings
from typing import Optional

from sklearn.cluster import KMeans as SklearnKmeans
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor, RandomForestClassifier, \
    RandomForestRegressor
from sklearn.linear_model import Lasso as SklearnLassoReg, LinearRegression as SklearnLinReg, \
    LogisticRegression as SklearnLogReg, Ridge as SklearnRidgeReg, SGDRegressor as SklearnSGD
from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN, \
    KNeighborsRegressor as SklearnKNNReg
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC as SklearnSVC, LinearSVR as SklearnSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB
from xgboost import XGBClassifier, XGBRegressor
from core.models.evaluation.custom_models.models import CustomSVC
from benchmark.tpot.b_tpot import fit_tpot, predict_tpot_class, predict_tpot_reg
from core.models.data import InputData, OutputData
from core.models.evaluation.automl_eval import fit_h2o, predict_h2o
from core.models.evaluation.hyperparams import params_range_by_model
from core.models.evaluation.lstm_eval import fit_lstm, predict_lstm
from core.models.evaluation.stats_models_eval import fit_ar, fit_arima, \
    predict_ar, predict_arima
from core.models.evaluation.data_evaluation_strategies import \
    get_difference, get_residual, get_sum, get_trend, get_data
from core.models.tuners import ForecastingCustomRandomTuner, \
    SklearnCustomRandomTuner, SklearnTuner

warnings.filterwarnings("ignore", category=UserWarning)


class EvaluationStrategy:
    def fit(self, train_data: InputData):
        raise NotImplementedError()

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        raise NotImplementedError()

    def fit_tuned(self, train_data: InputData, iterations: int = 30):
        raise NotImplementedError()


class SkLearnEvaluationStrategy(EvaluationStrategy):
    __model_by_types = {
        'xgboost': XGBClassifier,
        'xgbreg': XGBRegressor,
        'adareg': AdaBoostRegressor,
        'gbr': GradientBoostingRegressor,
        'knn': SklearnKNN,
        'knnreg': SklearnKNNReg,
        'dt': DecisionTreeClassifier,
        'dtreg': DecisionTreeRegressor,
        'treg': ExtraTreesRegressor,
        'rf': RandomForestClassifier,
        'rfr': RandomForestRegressor,
        'mlp': MLPClassifier,
        'lda': LinearDiscriminantAnalysis,
        'qda': QuadraticDiscriminantAnalysis,
        'linear': SklearnLinReg,
        'logit': SklearnLogReg,
        'ridge': SklearnRidgeReg,
        'lasso': SklearnLassoReg,
        'kmeans': SklearnKmeans,
        'svc': CustomSVC,
        'svr': SklearnSVR,
        'sgdr': SklearnSGD,
        'bernb': SklearnBernoulliNB
    }

    __metric_by_type = {
        'classification': make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True),
        'regression': make_scorer(mean_squared_error, greater_is_better=False),
    }

    def __init__(self, model_type: str):
        self._sklearn_model_impl = self._convert_to_sklearn(model_type)
        self._tune_func: SklearnTuner = Optional[SklearnTuner]
        self.params_for_fit = None
        self.model_type = model_type

    def fit(self, train_data: InputData):
        sklearn_model = self._sklearn_model_impl()
        sklearn_model.fit(train_data.features, train_data.target.ravel())
        return sklearn_model

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        raise NotImplementedError()

    def fit_tuned(self, train_data: InputData,
                  iterations: int = 30):
        trained_model = self.fit(train_data=train_data)
        params_range = params_range_by_model.get(self.model_type, None)
        metric = self.__metric_by_type.get(train_data.task.task_type.name, None)
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
            self.params_for_fit = tuned_params
            trained_model = self._sklearn_model_impl(**tuned_params)
            trained_model.fit(train_data.features, train_data.target.ravel())

        return trained_model, tuned_params

    def _convert_to_sklearn(self, model_type: str):
        if model_type in self.__model_by_types.keys():
            return self.__model_by_types[model_type]
        else:
            raise ValueError(f'Impossible to obtain SKlearn strategy for {model_type}')

    def _find_model_by_impl(self, impl):
        for model, model_impl in self.__model_by_types.items():
            if model_impl == impl:
                return model


class SkLearnClassificationStrategy(SkLearnEvaluationStrategy):
    def predict(self, trained_model, predict_data: InputData):
        n_classes = len(trained_model.classes_)
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


class StatsModelsForecastingStrategy(EvaluationStrategy):
    _model_functions_by_types = {
        'arima': (fit_arima, predict_arima),
        'ar': (fit_ar, predict_ar)
    }

    __default_params_by_model = {
        'arima': {'order': (2, 0, 0)},
        'ar': {'lags': (1, 2, 6, 12, 24)}
    }
    __params_range_by_model = {
        'arima': {'order': ((2, 0, 0), (5, 0, 5))},
        'ar': {'lags': (range(1, 6), range(6, 96, 6))}
    }

    def __init__(self, model_type: str):
        self._model_specific_fit, self._model_specific_predict = self._init_stats_model_functions(model_type)
        self._params_range = self.__params_range_by_model[model_type]
        self._default_params = self.__default_params_by_model[model_type]

        self.params_for_fit = None

    def _init_stats_model_functions(self, model_type: str):
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
        'tpot': (fit_tpot, predict_tpot_class),
        'h2o': (fit_h2o, predict_h2o)
    }

    def __init__(self, model_type: 'str'):
        self._model_specific_fit, self._model_specific_predict = \
            self._init_benchmark_model_functions(model_type)
        self.max_time_min = 5

    def _init_benchmark_model_functions(self, model_type):
        if model_type in self._model_functions_by_type.keys():
            return self._model_functions_by_type[model_type]
        else:
            raise ValueError(f'Impossible to obtain benchmark strategy for {model_type}')

    def fit(self, train_data: InputData):
        benchmark_model = self._model_specific_fit(train_data, self.max_time_min)
        return benchmark_model

    def predict(self, trained_model, predict_data: InputData):
        return self._model_specific_predict(trained_model, predict_data)

    def fit_tuned(self, train_data: InputData, iterations: int = 30):
        raise NotImplementedError()


class AutoMLRegressionStrategy(AutoMLEvaluationStrategy):
    _model_functions_by_type = {
        'tpot': (fit_tpot, predict_tpot_reg),
        'h2o': (fit_h2o, predict_h2o)
    }


# TODO inherit this and similar from custom strategy
class KerasForecastingStrategy(EvaluationStrategy):

    def __init__(self, model_type: str):
        self._init_lstm_model_functions(model_type)
        self.epochs = 10

    def _init_lstm_model_functions(self, model_type):
        if model_type != 'lstm':
            raise ValueError(f'Impossible to obtain forecasting strategy for {model_type}')

    def fit(self, train_data: InputData):
        model = fit_lstm(train_data, epochs=self.epochs)
        return model

    def predict(self, trained_model, predict_data: InputData):
        return predict_lstm(trained_model, predict_data)

    def tune(self, model, data_for_tune):
        raise NotImplementedError()


class DataModellingStrategy(EvaluationStrategy):
    _model_functions_by_type = {
        'direct_datamodel': get_data,
        'diff_data_model': get_difference,
        'additive_data_model': get_sum,
        'trend_data_model': get_trend,
        'residual_data_model': get_residual
    }

    def __init__(self, model_type: str):
        self._model_specific_predict = self._model_functions_by_type[model_type]

    def fit(self, train_data: InputData):
        # fit is not necessary for data models
        return None

    def predict(self, trained_model, predict_data: InputData):
        return self._model_specific_predict(predict_data)

    def fit_tuned(self, train_data: InputData, iterations: int = 30):
        return None, None
