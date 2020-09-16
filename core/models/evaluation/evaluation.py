import warnings
from abc import abstractmethod
from datetime import timedelta
from typing import Optional

from sklearn.cluster import KMeans as SklearnKmeans
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostRegressor,
                              ExtraTreesRegressor,
                              GradientBoostingRegressor,
                              RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import (Lasso as SklearnLassoReg,
                                  LinearRegression as SklearnLinReg,
                                  LogisticRegression as SklearnLogReg,
                                  Ridge as SklearnRidgeReg,
                                  SGDRegressor as SklearnSGD)
from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB
from sklearn.neighbors import (KNeighborsClassifier as SklearnKNN,
                               KNeighborsRegressor as SklearnKNNReg)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVR as SklearnSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from core.log import default_log, Log
from core.models.data import InputData, OutputData
from core.models.evaluation.custom_models.models import CustomSVC
from core.models.tuning.hyperparams import params_range_by_model
from core.models.tuning.tuners import SklearnTuner, SklearnCustomRandomTuner

warnings.filterwarnings("ignore", category=UserWarning)


class EvaluationStrategy:
    def __init__(self, model_type: str, params: Optional[dict] = None,
                 log=default_log(__name__)):
        self.params_for_fit = params
        self.model_type = model_type
        self.log: Log = log

    @abstractmethod
    def fit(self, train_data: InputData):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        raise NotImplementedError()

    def fit_tuned(self, train_data: InputData, iterations: int,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        raise NotImplementedError()

    @property
    def implementation_info(self) -> str:
        return 'No description'


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

    def __init__(self, model_type: str, params: Optional[dict] = None):
        self._sklearn_model_impl = self._convert_to_sklearn(model_type)
        self._tune_strategy: SklearnTuner = Optional[SklearnTuner]
        super().__init__(model_type, params)

    def fit(self, train_data: InputData):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            sklearn_model = self._sklearn_model_impl(**self.params_for_fit)
        else:
            sklearn_model = self._sklearn_model_impl()

        sklearn_model.fit(train_data.features, train_data.target.ravel())
        return sklearn_model

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        raise NotImplementedError()

    def fit_tuned(self, train_data: InputData, iterations: int,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        trained_model = self.fit(train_data=train_data)
        params_range = params_range_by_model.get(self.model_type, None)
        self._tune_strategy = SklearnCustomRandomTuner
        if not params_range:
            self.params_for_fit = None
            return trained_model, trained_model.get_params()

        tuned_params, best_model = self._tune_strategy(trained_model=trained_model,
                                                       tune_data=train_data,
                                                       params_range=params_range,
                                                       cross_val_fold_num=5,
                                                       time_limit=max_lead_time,
                                                       iterations=iterations).tune()

        if best_model or tuned_params:
            self.params_for_fit = tuned_params
            trained_model = best_model

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

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_sklearn(self.model_type))


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

    def fit_tuned(self, train_data: InputData, iterations: int = 30,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        raise NotImplementedError()
