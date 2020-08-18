from datetime import timedelta
from typing import Optional

import numpy as np
from ... import GARCH
from ... import VAR

from core.models.data import InputData, OutputData
from core.models.evaluation.evaluation import EvaluationStrategy
from core.models.tuning.tuners import ForecastingCustomRandomTuner


def fit_garch(train_data: InputData, params):
    return GARCH(train_data.target, **params).fit()


def fit_var(train_data: InputData, params):
    return VAR(train_data.target, **params).fit()


def predict_garch(trained_model, predict_data: InputData) -> OutputData:
#     start, end = trained_model.nobs, \
#                  trained_model.nobs + len(predict_data.target) - 1
    
    start, end = 0, len(predict_data.target)
    
    prediction = trained_model.predict(end)
    
    return prediction[0:len(predict_data.target)]


def predict_var(trained_model, predict_data: InputData) -> OutputData:
#     start, end = trained_model.nobs, \
#                  trained_model.nobs + len(predict_data.target) - 1
    
    start, end = 0, len(predict_data.target)
    
    prediction = trained_model.predict(end)
    
    return prediction[0:len(predict_data.target)]


class pyfluxModelsForecastingStrategy(EvaluationStrategy):
    __model_functions_by_types = {
        'garch': (fit_garch, predict_garch),
        'var': (fit_var, predict_var)
    }
    
    __model_description_by_func = {
        fit_var: 'pyflux.VAR',
        fit_garch: 'pyflux.GARCH'
    }
    __default_params_by_model = {
        'garch': {'p': (1, 1)},
        'var': {'q': (1, 1)}
    }
    __params_range_by_model = {
        'garch': {'p': ((1, 1), (2, 2))},
        'var': {'q': (range(1, 6), range(1, 6))}
    }

    def __init__(self, model_type: str, params: Optional[dict] = None):
        self._model_specific_fit, self._model_specific_predict = self._init_stats_model_functions(model_type)
        self._params_range = self.__params_range_by_model[model_type]
        self._default_params = self.__default_params_by_model[model_type]

        super().__init__(model_type, params)

    def _init_stats_model_functions(self, model_type: str):
        if model_type in self.__model_functions_by_types.keys():
            return self.__model_functions_by_types[model_type]
        else:
            raise ValueError(f'Impossible to obtain Stats strategy for {model_type}')

    def fit(self, train_data: InputData):
        stats_model = self._model_specific_fit(train_data, self._default_params)
        self.params_for_fit = self._default_params
        return stats_model

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        return self._model_specific_predict(trained_model, predict_data)

    def fit_tuned(self, train_data: InputData, iterations: int = 10,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        tuned_params = ForecastingCustomRandomTuner().tune(fit=self._model_specific_fit,
                                                           predict=self._model_specific_predict,
                                                           tune_data=train_data,
                                                           params_range=self._params_range,
                                                           default_params=self._default_params,
                                                           iterations=iterations)

        stats_model = self._model_specific_fit(train_data, tuned_params)
        self.params_for_fit = tuned_params

        return stats_model, tuned_params

    @property
    def implementation_info(self) -> str:
        return self.__model_description_by_func[self._model_specific_fit]
