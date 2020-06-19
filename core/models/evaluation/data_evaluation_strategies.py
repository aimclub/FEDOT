import numpy as np
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose

from core.models.data import InputData
from core.models.evaluation.evaluation import EvaluationStrategy
from core.repository.model_types_repository import ModelTypesIdsEnum


def get_data(predict_data: InputData):
    return predict_data.features


def get_difference(predict_data: InputData):
    number_of_inputs = predict_data.features.shape[1]
    if number_of_inputs != 1:
        raise ValueError('Too many inputs for the differential model')
    return predict_data.features[:, 0] - predict_data.target


def get_sum(predict_data: InputData):
    if predict_data.features.shape[1] != 2:
        raise ValueError('Wrong number of inputs for the additive model')
    return np.sum(predict_data.features, axis=1)


def _estimate_period(variable):
    analyse_ratio = 10
    f, pxx_den = signal.welch(variable, fs=1, scaling='spectrum',
                              nfft=int(len(variable) / analyse_ratio),
                              nperseg=int(len(variable) / analyse_ratio))
    period = int(1 / f[np.argmax(pxx_den)])
    return period


def get_trend(predict_data: InputData):
    target = predict_data.target
    period = _estimate_period(target)
    decomposed_target = seasonal_decompose(target, period=period, extrapolate_trend='freq')
    return decomposed_target.trend


def get_residual(predict_data: InputData):
    target_trend = get_trend(predict_data)
    target_residual = predict_data.target - target_trend
    return target_residual


class DataModellingStrategy(EvaluationStrategy):
    _model_functions_by_type = {
        ModelTypesIdsEnum.direct_datamodel: get_data,
        ModelTypesIdsEnum.diff_data_model: get_difference,
        ModelTypesIdsEnum.additive_data_model: get_sum,
        ModelTypesIdsEnum.trend_data_model: get_trend,
        ModelTypesIdsEnum.residual_data_model: get_residual
    }

    def __init__(self, model_type: ModelTypesIdsEnum):
        self._model_specific_predict = self._model_functions_by_type[model_type]

    def fit(self, train_data: InputData):
        # fit is not necessary for data models
        return None

    def predict(self, trained_model, predict_data: InputData):
        return self._model_specific_predict(predict_data)

    def tune(self, model, data_for_tune: InputData):
        return model
