import os

import pandas as pd
import numpy as np

from examples.decompose.refinement_forecast_example import time_series_into_input
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from sklearn.metrics import mean_absolute_error

from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast


def _get_lagged_refinement_pipeline(a_model: str = 'ridge'):
    """ Create 4-level pipeline with decompose operation """

    node_lagged = PrimaryNode('lagged')
    node_a = SecondaryNode(a_model, nodes_from=[node_lagged])

    node_decompose = SecondaryNode('decompose', nodes_from=[node_lagged, node_a])
    node_b = SecondaryNode('ridge', nodes_from=[node_decompose])

    # Combining branches with different targets (T and T_decomposed)
    node_c = SecondaryNode('ridge', nodes_from=[node_a, node_b])

    pipeline = Pipeline(node_c)
    return pipeline


def _get_lagged_non_refinement_pipeline(a_model: str = 'ridge'):
    """ Create 4-level pipeline without decompose operation """

    node_lagged = PrimaryNode('lagged')
    node_a = SecondaryNode(a_model, nodes_from=[node_lagged])
    node_b = SecondaryNode('ridge', nodes_from=[node_lagged])
    node_c = SecondaryNode('ridge', nodes_from=[node_a, node_b])

    pipeline = Pipeline(node_c)
    return pipeline


def _get_non_lagged_refinement_pipeline(a_model: str = 'arima'):
    """ Create pipeline with decompose operation where A model is 'non_lagged'
    but model B is require lagged transformation
    """

    node_a = PrimaryNode(a_model)
    node_lagged = PrimaryNode('lagged')

    node_decompose = SecondaryNode('decompose', nodes_from=[node_lagged, node_a])
    node_b = SecondaryNode('ridge', nodes_from=[node_decompose])

    # Combining branches with different targets (T and T_decomposed)
    node_c = SecondaryNode('ridge', nodes_from=[node_a, node_b])

    pipeline = Pipeline(node_c)
    return pipeline


def _get_non_lagged_non_refinement_pipeline(a_model: str = 'arima'):
    """
    Create pipeline without decompose operation where A model is 'non_lagged'
    """
    node_a = PrimaryNode(a_model)
    node_lagged = PrimaryNode('lagged')

    node_b = SecondaryNode('ridge', nodes_from=[node_lagged])
    node_c = SecondaryNode('ridge', nodes_from=[node_a, node_b])

    pipeline = Pipeline(node_c)
    return pipeline


def lstm_ref():
    return _get_non_lagged_refinement_pipeline('clstm')


def lstm_non_ref():
    return _get_non_lagged_non_refinement_pipeline('clstm')


def arima_ref():
    return _get_non_lagged_refinement_pipeline('arima')


def arima_non_ref():
    return _get_non_lagged_non_refinement_pipeline('arima')


def ridge_ref():
    return _get_lagged_refinement_pipeline('ridge')


def ridge_non_ref():
    return _get_lagged_non_refinement_pipeline('ridge')


def make_forecasts(time_series: np.array, ts_name: str, horizons: list,
                   tuner_iterations: int, validation_blocks: int = 3,
                   save_path: str = None):
    """ Launch forecasting based on the A model

    :param time_series: array with time series for forecasting
    :param ts_name: name of time series
    :param horizons: horizons for forecasting
    :param tuner_iterations: number of iterations for tuning
    :param validation_blocks: number of parts for time series validation
    :param save_path: folder to save predictions
    """
    ts_save_path = os.path.join(save_path, str(ts_name))
    if os.path.isdir(ts_save_path) is False:
        os.makedirs(ts_save_path)
    models_for_experiment = ['ridge', 'clstm', 'arima']
    pipelines_by_model = {'ridge': [ridge_ref, ridge_non_ref],
                          'clstm': [lstm_ref, lstm_non_ref],
                          'arima': [arima_ref, arima_non_ref]}

    # For every forecast horizon
    for len_forecast in horizons:
        print(f'Forecast for {len_forecast} elements')
        horizon = len_forecast * validation_blocks
        train_part = time_series[:-horizon]

        train_input, predict_input = time_series_into_input(len_forecast,
                                                            train_part,
                                                            time_series)

        # For every configuration
        for model in models_for_experiment:
            generators = pipelines_by_model.get(model)
            pipeline_ref = generators[0]()
            tuned_ref = pipeline_ref.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                                         input_data=train_input,
                                                         iterations=tuner_iterations,
                                                         timeout=2)
            predicted_ref = in_sample_ts_forecast(pipeline=tuned_ref,
                                                  input_data=predict_input,
                                                  horizon=horizon)

            pipeline_non_ref = generators[1]()
            tuned_non_ref = pipeline_non_ref.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                                                 input_data=train_input,
                                                                 iterations=tuner_iterations,
                                                                 timeout=2)
            predicted_non_ref = in_sample_ts_forecast(pipeline=tuned_non_ref,
                                                      input_data=predict_input,
                                                      horizon=horizon)

            time_series_decomposed = np.hstack((train_part, predicted_ref))
            time_series_non_decomposed = np.hstack((train_part, predicted_non_ref))
            df = pd.DataFrame({'actual': time_series, 'decomposed': time_series_decomposed,
                               'non_decompose': time_series_non_decomposed})

            df_name = ''.join(('forecast_', str(len_forecast), '_model_', str(model),
                               '_val_blocks_', str(validation_blocks), '.csv'))
            df.to_csv(os.path.join(ts_save_path, df_name), index=False)
