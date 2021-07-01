import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup

warnings.filterwarnings('ignore')


def plot_averaged_forecast(actual, predicted, first_column_act, last_column_act,
                           first_column_pr, last_column_pr):
    """
    Function plot averaged forecasts and actual time series

    :param actual: time series with averaged actual values
    :param predicted: time series with averaged predicted values
    :param first_column_act: numpy array with actual first days
    :param last_column_act: numpy array with actual seventh days
    :param first_column_pr: numpy array with predicted first days
    :param last_column_pr: numpy array with predicted seventh days
    """
    plt.plot(actual, label='7-day moving average actual')
    plt.fill_between(range(0, len(actual)), first_column_act, last_column_act, alpha=0.4)
    plt.plot(predicted, label='7-day moving average forecast')
    plt.fill_between(range(0, len(actual)), first_column_pr, last_column_pr, alpha=0.4)
    plt.ylabel('River level, cm', fontsize=14)
    plt.xlabel('Time index', fontsize=14)
    plt.grid()
    plt.legend(fontsize=12)
    plt.show()


def plot_last_day_forecast(last_column_act, last_column_pr):
    """ The function draws model predictions and actual water level values for
    every seventh day of the forecast (forecast horizon is 7 days)

    :param last_column_act: numpy array with actual seventh days
    :param last_column_pr: numpy array with predicted seventh days
    """
    plt.plot(last_column_act, label='7th day actual')
    plt.plot(last_column_pr, label='7th day forecast')
    plt.ylabel('River level, cm', fontsize=14)
    plt.xlabel('Time index', fontsize=14)
    plt.grid()
    plt.legend(fontsize=12)
    plt.show()


def plot_predictions(predicted_array, test_output):
    """ Function plot the predictions of the algorithm """
    predicted_columns = np.array(predicted_array)
    actual_columns = np.array(test_output.target)

    # Take mean value for columns and also first and last columns
    predicted = predicted_columns.mean(axis=1)
    first_column_pr = np.ravel(predicted_columns[:, 0])
    last_column_pr = np.ravel(predicted_columns[:, -1])
    actual = actual_columns.mean(axis=1)
    first_column_act = np.ravel(actual_columns[:, 0])
    last_column_act = np.ravel(actual_columns[:, -1])

    plot_averaged_forecast(actual, predicted, first_column_act,
                           last_column_act,
                           first_column_pr, last_column_pr)

    plot_last_day_forecast(last_column_act, last_column_pr)


def run_multi_output_case(path, vis=False):
    """ Function launch case for river levels prediction on Lena river as
    multi-output regression task

    :param path: path to the file with table
    :param vis: is it needed to visualise pipeline and predictions
    """
    target_columns = ['1_day', '2_day', '3_day', '4_day', '5_day', '6_day', '7_day']

    data = InputData.from_csv(path, target_columns=target_columns, columns_to_drop=['date'])
    train, test = train_test_data_setup(data)

    problem = 'regression'

    automl_model = Fedot(problem=problem)
    automl_model.fit(features=train)
    predicted_array = automl_model.predict(features=test)

    # Convert output into one dimensional array
    forecast = np.ravel(predicted_array)

    mae_value = mean_absolute_error(np.ravel(test.target), forecast)
    print(f'MAE - {mae_value:.2f}')

    if vis:
        plot_predictions(predicted_array, test)


if __name__ == '__main__':
    path_file = './data/lena_levels/multi_sample.csv'
    run_multi_output_case(path_file, vis=True)
