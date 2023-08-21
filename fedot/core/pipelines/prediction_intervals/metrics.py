import numpy as np


def quantile_loss(y_true: np.array, y_pred: np.array, quantile: float):
    """Qauntile loss of Pinball loss function."""

    res = y_true - y_pred
    metrics = np.empty(shape=[0])
    for x in res:
        if x >= 0:
            metrics = np.append(metrics, quantile * x)
        else:
            metrics = np.append(metrics, (quantile - 1) * x)

    return np.mean(metrics)


def picp(true: np.array, low: np.array, up: np.array):
    """Prediction interval coverage probability metric. See, e.g., https://www.mdpi.com/2673-4826/2/1/2"""

    s = 0
    length = len(true)
    for i in range(length):
        if true[i] >= low[i] and true[i] <= up[i]:
            s += 1

    return s / length


def interval_score(true: np.array, low: np.array, up: np.array, alpha: float = 0.1, weight: float = 2):
    """Interval score metric, see https://arxiv.org/pdf/2007.05709.pdf"""

    true = np.array(true)
    low = np.array(low)
    up = np.array(up)
    aiw = (up - low).mean()
    array = np.array([])
    length = len(true)
    for i in range(length):
        if true[i] < low[i]:
            array = np.append(array, low[i] - true[i])
        elif true[i] > up[i]:
            array = np.append(array, true[i] - up[i])
        else:
            array = np.append(array, 0)

    return aiw + weight / alpha * array.mean()
