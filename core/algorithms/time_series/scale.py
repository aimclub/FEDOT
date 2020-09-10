import numpy as np
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose

from core.models.data import InputData


def split_ts_to_components(trained_model, predict_data: InputData):
    """
    :param trained_model: in this case it is the value of period obtained during fitting
    :param predict_data: dataset that contains time series in target
    :return: extracted trend
    """

    target = predict_data.target
    period = trained_model

    # TODO implement better decomposition, now it is workaround for 'x must have 2 complete cycles' error
    if period * 2 >= len(target):
        period = round(len(target) / 2)

    decomposed_target = seasonal_decompose(target, period=period, extrapolate_trend='freq')
    trend = decomposed_target.trend
    residual = predict_data.target - trend

    return trend, residual


def merge_component_with_exog(component, predict_data: InputData):
    if predict_data.features is not None and not np.array_equal(predict_data.target, predict_data.features):
        component = component[:, None]
        component = np.concatenate((component, predict_data.features), axis=1)
    return component


def estimate_period(variable):
    analyse_ratio = 10
    f, pxx_den = signal.welch(variable, fs=1, scaling='spectrum',
                              nfft=int(len(variable) / analyse_ratio),
                              nperseg=int(len(variable) / analyse_ratio))
    period = int(1 / f[np.argmax(pxx_den)])
    return period
