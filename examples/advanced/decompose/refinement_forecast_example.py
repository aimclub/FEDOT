import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams

warnings.filterwarnings('ignore')
np.random.seed(2020)


def get_refinement_pipeline_with_polyfit():
    """ Create 4-level pipeline with decompose operation """

    node_polyfit = PrimaryNode('polyfit')
    node_polyfit.parameters = {'degree': 2}
    node_lagged = PrimaryNode('lagged')
    node_decompose = SecondaryNode('decompose', nodes_from=[node_lagged, node_polyfit])
    node_dtreg = SecondaryNode('dtreg', nodes_from=[node_decompose])
    node_dtreg.parameters = {'max_depth': 3}

    # Pipelines with different outputs
    pipeline_with_decompose_finish = Pipeline(node_dtreg)
    pipeline_with_main_finish = Pipeline(node_polyfit)

    # Combining branches with different targets (T and T_decomposed)
    final_node = SecondaryNode('ridge', nodes_from=[node_polyfit, node_dtreg])

    pipeline = Pipeline(final_node)
    return pipeline_with_main_finish, pipeline_with_decompose_finish, pipeline


def get_refinement_pipeline(lagged):
    """ Create 4-level pipeline with decompose operation """

    node_lagged = PrimaryNode('lagged')
    node_lagged.parameters = {'window_size': lagged}
    node_lasso = SecondaryNode('lasso', nodes_from=[node_lagged])
    node_decompose = SecondaryNode('decompose', nodes_from=[node_lagged, node_lasso])
    node_dtreg = SecondaryNode('dtreg', nodes_from=[node_decompose])
    node_dtreg.parameters = {'max_depth': 3}

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
    node_lagged.parameters = {'window_size': lagged}
    node_lasso = SecondaryNode('lasso', nodes_from=[node_lagged])
    node_dtreg = SecondaryNode('dtreg', nodes_from=[node_lagged])
    node_dtreg.parameters = {'max_depth': 3}
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
    mse = mean_squared_error(test_part.target, predicted_values, squared=False)
    mae = mean_absolute_error(test_part.target, predicted_values)
    print(f'RMSE {pipeline_name} - {mse:.4f}')
    print(f'MAE {pipeline_name} - {mae:.4f}\n')


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
    idx = np.array(pd.to_datetime(df['datetime']))
    # 3 folds for validation
    horizon = len_forecast * validation_blocks

    # Get pipeline with decomposing operation
    pipeline_with_main_finish, pipeline_with_decompose_finish, pipeline = get_refinement_pipeline(lagged)
    # Get simple pipeline without decomposing operation
    simple_pipeline = get_non_refinement_pipeline(lagged)

    train_input, predict_input = train_test_data_setup(
        InputData(idx=idx,
                  features=time_series,
                  target=time_series,
                  task=Task(TaskTypesEnum.ts_forecasting,
                            TsForecastingParams(
                                forecast_length=horizon)),
                  data_type=DataTypesEnum.ts))

    # Forecast of pipeline with decomposition
    predicted_values = in_sample_fit_predict(pipeline, train_input,
                                             predict_input, horizon)
    display_metrics(predict_input, predicted_values, pipeline_name='with decomposition')

    # Range for visualisation
    ids_for_test = idx[range(len(train_input.target), len(time_series))]
    plt.plot(idx, time_series, label='Actual time series')
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
        display_metrics(predict_input, predicted_simple, pipeline_name='without decomposition')

    i = len(train_input.target)
    for _ in range(0, validation_blocks):
        deviation = np.std(predicted_values)
        plt.plot([idx[i], idx[i]], [min(predicted_values) - deviation, max(predicted_values) + deviation],
                 c='black', linewidth=1)
        i += len_forecast
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    path = '../../../cases/data/time_series/economic_data.csv'
    run_refinement_forecast(path, len_forecast=50, validation_blocks=5,
                            lagged=50, vis_with_decompose=True)
