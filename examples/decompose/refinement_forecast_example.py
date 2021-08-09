import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from examples.time_series.ts_forecasting_tuning import prepare_input_data
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum

warnings.filterwarnings('ignore')
np.random.seed(2020)


def get_refinement_pipeline(lagged):
    """ Create 4-level pipeline with decompose operation """

    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': lagged}
    node_lasso = SecondaryNode('lasso', nodes_from=[node_lagged])
    node_decompose = SecondaryNode('decompose', nodes_from=[node_lagged, node_lasso])
    node_dtreg = SecondaryNode('dtreg', nodes_from=[node_decompose])
    node_dtreg.custom_params = {'max_depth': 3}

    # Pipelines with different outputs
    pipeline_with_decompose_finish = Pipeline(node_dtreg)
    pipeline_with_main_finish = Pipeline(node_lasso)

    # Combining branches with different targets (T and T_decomposed)
    final_node = SecondaryNode('ridge', nodes_from=[node_lasso, node_dtreg])

    pipeline = Pipeline(final_node)
    return pipeline_with_main_finish, pipeline_with_decompose_finish, pipeline


def get_non_refinement_pipeline(lagged):
    """ Create 4-level pipeline without decompose operation """

    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': lagged}
    node_lasso = SecondaryNode('lasso', nodes_from=[node_lagged])
    node_dtreg = SecondaryNode('dtreg', nodes_from=[node_lagged])
    node_dtreg.custom_params = {'max_depth': 3}
    final_node = SecondaryNode('ridge', nodes_from=[node_lasso, node_dtreg])

    pipeline = Pipeline(final_node)
    return pipeline


def in_sample_fit_predict(pipeline, train_input, predict_input, horizon) -> np.array:
    """ Fit pipeline and make predictions (in-sample forecasting) """
    pipeline.fit(train_input)

    predicted_main = in_sample_ts_forecast(pipeline=pipeline,
                                           input_data=predict_input,
                                           horizon=horizon)
    return predicted_main


def display_metrics(test_part, predicted_values, pipeline_name):
    mse = mean_squared_error(test_part, predicted_values, squared=False)
    mae = mean_absolute_error(test_part, predicted_values)
    print(f'RMSE {pipeline_name} - {mse:.4f}')
    print(f'MAE {pipeline_name} - {mae:.4f}\n')


def time_series_into_input(len_forecast, train_part, time_series):
    """ Function wrap univariate time series into InputData """
    train_input, _, task = prepare_input_data(len_forecast=len_forecast,
                                              train_data_features=train_part,
                                              train_data_target=train_part,
                                              test_data_features=train_part)
    # Create data for validation
    predict_input = InputData(idx=range(0, len(time_series)),
                              features=time_series,
                              target=time_series,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input


def run_refinement_forecast(path_to_file, len_forecast=100, lagged=150,
                            validation_blocks=3, vis_with_decompose=True):
    """ Function launch example with experimental features of the FEDOT framework

    :param path_to_file: path to the csv file
    :param len_forecast: forecast length
    :param lagged: window size for lagged transformation
    :param validation_blocks: amount of parts for time series validation
    :param vis_with_decompose: visualise part of main forecast
    """

    # Read dataframe
    df = pd.read_csv(path_to_file)
    time_series = np.array(df['value'])
    # 3 folds for validation
    horizon = len_forecast*validation_blocks
    train_part = time_series[:-horizon]
    test_part = time_series[-horizon:]

    # Get pipeline with decomposing operation
    pipeline_with_main_finish, pipeline_with_decompose_finish, pipeline = get_refinement_pipeline(lagged)
    # Get simple pipeline without decomposing operation
    simple_pipeline = get_non_refinement_pipeline(lagged)

    train_input, predict_input = time_series_into_input(len_forecast,
                                                        train_part,
                                                        time_series)

    # Forecast of pipeline with decomposition
    predicted_values = in_sample_fit_predict(pipeline, train_input,
                                             predict_input, horizon)
    display_metrics(test_part, predicted_values, pipeline_name='with decomposition')

    # Range for visualisation
    ids_for_test = range(len(train_part), len(time_series))
    plt.plot(time_series, label='Actual time series')
    plt.plot(ids_for_test, predicted_values, label='With decomposition')

    if vis_with_decompose:
        # Forecast of first model in the pipeline
        predicted_main = in_sample_fit_predict(pipeline_with_main_finish, train_input,
                                               predict_input, horizon)
        # Forecast for residuals
        predicted_decompose = in_sample_fit_predict(pipeline_with_decompose_finish, train_input,
                                                    predict_input, horizon)

        plt.plot(ids_for_test, predicted_main, label='Main branch forecast')
        plt.plot(ids_for_test, predicted_decompose, label='Residual branch forecast')
    else:
        predicted_simple = in_sample_fit_predict(simple_pipeline, train_input,
                                                 predict_input, horizon)
        plt.plot(ids_for_test, predicted_simple, label='Non decomposition')
        display_metrics(test_part, predicted_simple, pipeline_name='without decomposition')

    i = len(train_part)
    for _ in range(0, validation_blocks):
        deviation = np.std(predicted_values)
        plt.plot([i, i], [min(predicted_values)-deviation, max(predicted_values)+deviation],
                 c='black', linewidth=1)
        i += len_forecast

    plt.legend()
    plt.show()


if __name__ == '__main__':
    path = '../../cases/data/time_series/economic_data.csv'
    run_refinement_forecast(path, len_forecast=50, validation_blocks=5,
                            lagged=50, vis_with_decompose=False)
