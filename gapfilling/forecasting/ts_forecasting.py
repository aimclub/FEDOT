import os
import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from matplotlib import pyplot as plt
import timeit
from pylab import rcParams
rcParams['figure.figsize'] = 18, 7
import warnings
warnings.filterwarnings('ignore')
from gapfilling.validation_and_metrics import *

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def make_forecast(chain, train_data, len_forecast: int, max_window_size: int):
    """
    Function for predicting values in a time series

    :param chain: TsForecastingChain object
    :param train_data: one-dimensional numpy array to train chain
    :param len_forecast: amount of values for predictions
    :param max_window_size: moving window size

    :return predicted_values: numpy array, forecast of model
    """

    # Here we define which task should we use, here we also define two main
    # hyperparameters: forecast_length and max_window_size
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(train_data)),
                            features=None,
                            target=train_data,
                            task=task,
                            data_type=DataTypesEnum.ts)

    # Make a "blank", here we need just help FEDOT understand that the
    # forecast should be made exactly the "len_forecast" length
    predict_input = InputData(idx=np.arange(0, len_forecast),
                              features=None,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    # Fit it
    chain.fit_from_scratch(train_input)

    # Predict
    predicted_values = chain.forecast(initial_data=train_input,
                                      supplementary_data=predict_input).predict

    return predicted_values


def plot_results(actual_time_series, predicted_values, len_train_data,
                 y_name='Parameter'):
    """
    Function for drawing plot with predictions

    :param actual_time_series: the entire array with one-dimensional data
    :param predicted_values: array with predicted values
    :param len_train_data: number of elements in the training sample
    :param y_name: name of the y axis
    """

    plt.plot(np.arange(0, len(actual_time_series)),
             actual_time_series, label='Actual values', c='green')
    plt.plot(np.arange(len_train_data, len_train_data + len(predicted_values)),
             predicted_values, label='Predicted', c='blue')
    # Plot black line which divide our array into train and test
    plt.plot([len_train_data, len_train_data],
             [min(actual_time_series), max(actual_time_series)], c='black',
             linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.show()


def plot_double_results(original_time_series, actual_time_series,
                        predicted_values, len_train_data,
                        y_name='Parameter'):
    """
    Function for drawing plot with predictions

    :param original_time_series: original time series without gaps
    :param actual_time_series: the entire array with one-dimensional data
    (recovered data)
    :param predicted_values: array with predicted values
    :param len_train_data: number of elements in the training sample
    :param y_name: name of the y axis
    """

    plt.plot(np.arange(0, len(original_time_series)),
             original_time_series, label='Original time series', c='green')
    plt.plot(np.arange(0, len(actual_time_series)),
             actual_time_series, label='Recovered time series', c='blue')
    plt.plot(np.arange(len_train_data, len_train_data + len(predicted_values)),
             predicted_values, label='Forecast', c='red')
    # Plot black line which divide our array into train and test
    plt.plot([len_train_data, len_train_data],
             [min(actual_time_series), max(actual_time_series)], c='black',
             linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.show()


def get_chain():

    # Initialization of the chain
    forecasting_chain = TsForecastingChain()
    node_1_first = PrimaryNode('ridge')
    node_1_second = PrimaryNode('ridge')

    node_2_first = SecondaryNode('linear', nodes_from=[node_1_first])
    node_2_second = SecondaryNode('linear', nodes_from=[node_1_second])
    node_final = SecondaryNode('svr', nodes_from=[node_2_first,
                                                  node_2_second])
    forecasting_chain.add_node(node_final)

    return forecasting_chain

def run_forecasting_task(files_list, folders_list, columns_with_gap,
                         len_forecast, max_window_size, vis = True):
    """
    The function starts the algorithm of gap-filling

    :param files_list: where to save csv files with filled gaps
    :param folders_list: list with paths with filled time series
    :param columns_with_gap: list with names of columns with gaps
    :param len_forecast: forecast length in time series elements
    :param max_window_size: moving window size
    :param vis: is there a need to make visualisations
    """

    # Read every time series for processing
    for file in files_list:
        source_data = pd.read_csv(os.path.join('..', 'data', file))
        source_data['Date'] = pd.to_datetime(source_data['Date'])

        ########################
        # Original forecasting #
        ########################
        original_time_series = np.array(source_data['Height'])
        # Train test split
        train_part = original_time_series[:-len_forecast]
        test_part = original_time_series[-len_forecast:]

        # Get chain to make predictions
        forecasting_chain = get_chain()

        # Make forecast
        predicted_vals = make_forecast(forecasting_chain, train_part,
                                       len_forecast, max_window_size)

        # Check forecast errors
        mape_orig = mean_absolute_percentage_error(test_part, predicted_vals)
        print(f'\nOriginal time series MAPE - {mape_orig:.2f}')

        if vis == True:
            name = ''.join(('Original time series from file ', file))
            plot_results(original_time_series, predicted_vals, len(train_part),
                         y_name=name)

        # For particular time series
        for type_of_gap in columns_with_gap:
            print(f'\nFile name {file}. Type of the recovered gap - {type_of_gap}')

            for current_folder in folders_list:
                path_to_file = os.path.join(current_folder, file)
                modif_data = pd.read_csv(path_to_file)
                modif_data['Date'] = pd.to_datetime(modif_data['Date'])

                # get folder name
                gapfill_alg = os.path.basename(current_folder)

                # This column was recovered using gapfill_alg
                recovered_time_series = np.array(modif_data[type_of_gap])

                # Train split
                train_rec_part = recovered_time_series[:-len_forecast]

                # Make forecast
                forecasting_chain = get_chain()
                predicted_vals = make_forecast(forecasting_chain, train_rec_part,
                                               len_forecast, max_window_size)

                mape_rec = mean_absolute_percentage_error(test_part,
                                                          predicted_vals)
                print(f'Recovered time series MAPE. {gapfill_alg} - {mape_rec:.2f}\n')

                if vis == True:
                    name = ''.join(('Recovered time series. ', gapfill_alg))
                    plot_double_results(original_time_series,
                                        recovered_time_series,
                                        predicted_vals,
                                        len(train_rec_part),
                                        y_name=name)

# Run forecasting validation algprithm
files_list = ['Synthetic.csv', 'Sea_hour.csv', 'Sea_10_240.csv']
folders_list = ['../data/linear', '../data/poly', '../data/fedot_ridge']
columns_with_gap = ['gap', 'gap_center']
len_forecast = 400
max_window_size = 150

if __name__ == '__main__':
    run_forecasting_task(files_list, folders_list, columns_with_gap,
                         len_forecast, max_window_size,
                         vis=True)
