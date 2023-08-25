import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_intervals(model_forecast: np.array,
                              up_int: np.array,
                              low_int: np.array,
                              ts: np.array,
                              show_history: bool,
                              ts_test: np.array,
                              show_forecast: bool,
                              labels: str,
                              figwidth: float = 15,
                              figheight: float = 7):
    """Function for plotting prediction intervals.

    Args:
        model_forecast: forecast of the given Fedot class object
        up_int: upper prediction interval
        low_int: low prediction interval
        ts: initial time series
        show_history: flag whether to plot initial time series
        ts_test: test time series if given
        show_forecast: flag wheter to plot forecast of the given Fedot class object
        labels: which type of labels (prediction intervals or base quantiles) use for plotting.
    """
    if ts_test is not None:
        if len(ts_test) != len(model_forecast):
            raise ValueError('Lengths of test series and forecasting horizon are different. Correct test series.')

    train_len = len(ts)
    train_range = range(train_len)
    test_len = len(model_forecast)
    test_range = range(train_len, train_len + test_len)

    fig, ax = plt.subplots()
    fig.set(figwidth=figwidth, figheight=figheight)

    if labels == 'pred_ints':
        label = 'PredictionIntervals'
    elif labels == 'base_quantiles':
        label = 'Base quantiles'

    if ts_test is not None:
        ax.plot(test_range, ts_test, color='black', label='Actual TS')
    if show_forecast:
        ax.plot(test_range, model_forecast, color='red', label='Forecast')
    ax.fill_between(test_range, low_int, up_int, alpha=0.2, color='red', label=label)

    plt.legend()
    plt.grid()

    if show_history:

        fig1, ax1 = plt.subplots()
        fig1.set(figwidth=figwidth, figheight=figheight)
        ax1.plot(train_range, ts, color='black', alpha=0.5, label='Train TS')

        if ts_test is not None:
            ax1.plot(test_range, ts_test, color='black', label='Actual TS')
        if show_forecast:
            ax1.plot(test_range, model_forecast, color='red', label='Forecast')
        ax1.fill_between(test_range, low_int, up_int, alpha=0.2, color='red', label=label)

    plt.legend()
    plt.grid()


def _plot_prediction_intervals(horizon,
                               up_predictions,
                               low_predictions,
                               predictions,
                               model_forecast,
                               up_int,
                               low_int,
                               ts,
                               show_up_int=True,
                               show_low_int=True,
                               show_forecast=True,
                               show_history=True,
                               show_up_train=True,
                               show_low_train=True,
                               show_train=True,
                               ts_test=None):
    """Function for plotting prediction intervals. Used for developing, will be removed."""

    r = range(1, horizon + 1)
    if up_predictions is not None and low_predictions is not None:
        length = len(up_predictions)
    elif predictions is not None:
        length = len(predictions)

    fig, ax = plt.subplots()
    fig.set(figwidth=15, figheight=7)

    for i in range(length):
        if i == 0:
            if show_up_train and up_predictions is not None:
                ax.plot(r, up_predictions[i], color='yellow', label='predictions to build up interval')
            if show_low_train and low_predictions is not None:
                ax.plot(r, low_predictions[i], color='pink', label='predictions to build low interval')
            if show_train and predictions is not None:
                ax.plot(r, predictions[i], color='pink', label='predictions to build intervals')
        else:
            if show_up_train and up_predictions is not None:
                ax.plot(r, up_predictions[i], color='yellow')
            if show_low_train and low_predictions is not None:
                ax.plot(r, low_predictions[i], color='pink')
            if show_train and predictions is not None:
                ax.plot(r, predictions[i], color='pink')
    if show_up_int:
        ax.plot(r, up_int, color='blue', label='Up', marker='.')
    if show_low_int:
        ax.plot(r, low_int, color='green', label='Low', marker='.')
    if show_forecast:
        ax.plot(r, model_forecast, color='red', label='Forecast')
    if ts_test is not None:
        ax.plot(r, ts_test, color='black', label='Actual TS')
    plt.legend()

    if show_history:
        fig1, ax1 = plt.subplots()
        fig1.set(figwidth=15, figheight=7)
        train_range = range(len(ts))
        test_range = range(len(ts), len(ts) + horizon)

        ax1.plot(train_range, ts, color='gray', label='Train ts')
        ax1.plot(test_range, up_int, color='blue', label='Up')
        ax1.plot(test_range, low_int, color='green', label='Low')
        ax1.plot(test_range, model_forecast, color='red', label='Forecast')
        if ts_test is not None:
            ax1.plot(test_range, ts_test, color='black', label='Actual TS')
    plt.legend()
