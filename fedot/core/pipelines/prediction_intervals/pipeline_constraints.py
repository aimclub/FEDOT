import numpy as np

from fedot.core.pipelines.prediction_intervals.utils import ts_jumps, ts_deviance


def first_prediction_constraint(ts_train: np.array, forecast: np.array, prediction: np.array):
    """Function to check whether difference between first elements of pipeline prediction and forecast is not too big.

    This function computes difference between the first elements of some pipeline prediction and given model forecast.
    If this difference is bigger than maximal upswing of ts_train or smaller than minimal downswing of ts_train, then
    pipeline assumed to be unreliable and function returns False.

    Args:
        ts_train: initial time series
        forecast: forecast of given Fedot class object
        prediction: a forecast obtained by some pipeline.

    Returns:
        bool: True if difference between first elements of pipeline prediction and forecast is not too big, otherwise
              False.
    """
    return ts_jumps(ts_train)['low'] <= prediction[0] - forecast[0] <= ts_jumps(ts_train)['up']


def deviance_constraint(ts_train: np.array, prediction: np.array, constraint=2):
    """"Function to check whether a pipeline prediction oscillate to much comparing with train series.

    Args:
        ts_train: initial time series
        prediction: a forecast obtained by some mutation
        constraint: strength of oscillation restriction.

    Returns:
        True if prediction does not oscillate too much, otherwise False.
    """
    return ts_deviance(prediction) <= constraint * ts_deviance(ts_train)
