import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose

from core.models.data import InputData
from core.models.evaluation.evaluation import EvaluationStrategy

DIM_REDUCTION_EXPLAINED_VARIANCE_THR = 0.9
DIM_REDUCTION_MIN_EXPLAINED_VARIANCE = 0.01


def _estimate_period(variable):
    analyse_ratio = 10
    f, pxx_den = signal.welch(variable, fs=1, scaling='spectrum',
                              nfft=int(len(variable) / analyse_ratio),
                              nperseg=int(len(variable) / analyse_ratio))
    period = int(1 / f[np.argmax(pxx_den)])
    return period


def get_data(trained_model, predict_data: InputData):
    return predict_data.features


def get_difference(trained_model, predict_data: InputData):
    number_of_inputs = predict_data.features.shape[1]
    if number_of_inputs != 1:
        raise ValueError('Too many inputs for the differential model')
    return predict_data.features[:, 0] - predict_data.target


def get_sum(trained_model, predict_data: InputData):
    if predict_data.features.shape[1] != 2:
        raise ValueError('Wrong number of inputs for the additive model')
    return np.sum(predict_data.features, axis=1)


def fit_trend(train_data: InputData):
    target = train_data.target
    period = _estimate_period(target)
    return period


def get_trend(trained_model, predict_data: InputData):
    target = predict_data.target
    period = trained_model
    decomposed_target = seasonal_decompose(target, period=period, extrapolate_trend='freq')
    return decomposed_target.trend


def fit_residual(train_data: InputData):
    return fit_trend(train_data)


def get_residual(trained_model, predict_data: InputData):
    target_trend = get_trend(trained_model, predict_data)
    target_residual = predict_data.target - target_trend
    return target_residual


def fit_pca(train_data: InputData):
    pca = PCA(svd_solver='randomized', iterated_power='auto')
    pca.fit(train_data.features)
    return pca


def predict_pca(pca_model, predict_data: InputData):
    cum_variance = np.cumsum(pca_model.explained_variance_ratio_)
    last_ind_cum = min(np.where(cum_variance > DIM_REDUCTION_EXPLAINED_VARIANCE_THR)[0])
    last_ind = min(np.where(pca_model.explained_variance_ratio_ < DIM_REDUCTION_MIN_EXPLAINED_VARIANCE)[0])

    return pca_model.transform(predict_data.features)[:, :min(last_ind, last_ind_cum)]


class DataModellingStrategy(EvaluationStrategy):
    _model_functions_by_type = {
        'direct_data_model': (None, get_data),
        'diff_data_model': (None, get_difference),
        'additive_data_model': (None, get_sum),
        'trend_data_model': (fit_residual, get_trend),
        'residual_data_model': (fit_residual, get_residual),
        'pca_data_model': (fit_pca, predict_pca)
    }

    def __init__(self, model_type: str):
        self._model_specific_fit = self._model_functions_by_type[model_type][0]
        self._model_specific_predict = self._model_functions_by_type[model_type][1]

    def fit(self, train_data: InputData):
        if not self._model_specific_fit:
            return None
        else:
            return self._model_specific_fit(train_data)

    def predict(self, trained_model, predict_data: InputData):
        return self._model_specific_predict(trained_model, predict_data)

    def fit_tuned(self, **args):
        return None, None
