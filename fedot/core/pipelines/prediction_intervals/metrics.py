import numpy as np


def quantile_loss(y_true: np.array, y_pred: np.array, quantile: float):
    """Qauntile loss of Pinball loss function."""

    res = y_true - y_pred
    res[res >= 0] = res[res >= 0] * quantile
    res[res < 0] = res[res < 0] * (quantile - 1)
    return np.mean(res)


def picp(true: np.array, low: np.array, up: np.array):
    """Prediction interval coverage probability metric. See, e.g., https://www.mdpi.com/2673-4826/2/1/2"""
    return np.mean((true >= low) & (true <= up))


def interval_score(true: np.array, low: np.array, up: np.array, alpha: float = 0.1, weight: float = 2):
    """Interval score metric, see https://arxiv.org/pdf/2007.05709.pdf"""
    aiw = np.mean(up - low)
    array = (true < low) * (low - true) + (true > up) * (true - up)
    return aiw + weight / alpha * np.mean(array)
