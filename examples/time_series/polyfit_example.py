import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from examples.time_series.ts_forecasting_tuning import prepare_input_data
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


def polyfit_pipeline(degree):
    node_polyfit = PrimaryNode('polyfit')
    node_polyfit.custom_params = {"degree": degree}
    return Pipeline(node_polyfit)


def polyfit_ridge_pipeline(degree):
    node_polyfit = PrimaryNode('polyfit')
    node_polyfit.custom_params = {"degree": degree}
    node_lagged = PrimaryNode('lagged')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    node_ridge1 = SecondaryNode('ridge', nodes_from=[node_ridge, node_polyfit])

    return Pipeline(node_ridge1)


def run_experiment_with_polyfit(time_series, len_forecast=250,
                                degree=2):
    """ Function with example how time series trend could be approximated by a polynomial function
    for next extrapolation. Try different degree params to see a difference.

    :param time_series: time series for prediction
    :param len_forecast: forecast length
    :param degree: degree of polynomial function
    """

    # Let's divide our data on train and test samples
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    # Source time series
    train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                          train_data_features=train_data,
                                                          train_data_target=train_data,
                                                          test_data_features=train_data)

    pipeline = polyfit_pipeline(degree)
    pipeline.fit(train_input)

    predict = np.ravel(np.array(pipeline.predict(predict_input).predict))
    test_data = np.ravel(test_data)

    mse_before = mean_squared_error(test_data, predict, squared=False)
    mae_before = mean_absolute_error(test_data, predict)
    print(f'RMSE before tuning - {mse_before:.4f}')
    print(f'MAE before tuning - {mae_before:.4f}\n')

    pipeline.print_structure()
    plt.plot(range(0, len(time_series)), time_series, label='Actual time series')
    plt.plot(range(len(train_data), len(time_series)), predict, label='Forecast with polyfit')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../data/beer.csv')
    time_series = np.array(df.iloc[:, -1])
    run_experiment_with_polyfit(time_series,
                                len_forecast=50,
                                degree=2)
