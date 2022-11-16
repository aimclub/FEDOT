import warnings
from copy import copy
from datetime import timedelta

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

warnings.filterwarnings('ignore')


def run_river_experiment(file_path, pipeline, iterations=20, tuner=None,
                         tuner_iterations=100):
    """ Function launch experiment for river level prediction. Tuner processes
    are available for such experiment.

    :param file_path: path to the file with river level data
    :param pipeline: pipeline to fit and make prediction
    :param iterations: amount of iterations to process
    :param tuner: if tuning after composing process is required or not. tuner -
    NodesTuner or PipelineTuner.
    :param tuner_iterations: amount of iterations for tuning
    """

    # Read dataframe and prepare train and test data
    data = InputData.from_csv(file_path, target_columns='level_station_2',
                              task=Task(TaskTypesEnum.regression),
                              columns_to_drop=['date'])
    train_input, predict_input = train_test_data_setup(data)
    y_data_test = np.array(predict_input.target)

    for i in range(0, iterations):
        print(f'Iteration {i}\n')

        # To avoid inplace transformations make copy
        current_pipeline = copy(pipeline)

        # Fit it
        current_pipeline.fit(train_input)

        # Predict
        predicted_values = current_pipeline.predict(predict_input)
        preds = predicted_values.predict

        y_data_test = np.ravel(y_data_test)
        mse_value = mean_squared_error(y_data_test, preds, squared=False)
        mae_value = mean_absolute_error(y_data_test, preds)

        print(f'Obtained metrics for current iteration {i}:')
        print(f'RMSE - {mse_value:.2f}')
        print(f'MAE - {mae_value:.2f}\n')

        if tuner is not None:
            print('Start tuning process ...')
            pipeline_tuner = TunerBuilder(data.task)\
                .with_tuner(tuner)\
                .with_metric(RegressionMetricsEnum.MAE)\
                .with_iterations(tuner_iterations)\
                .with_timeout(timedelta(seconds=30)) \
                .build(train_input)
            tuned_pipeline = pipeline_tuner.tune(current_pipeline)

            # Fit it
            tuned_pipeline.fit(train_input)

            # Predict
            predicted_values_tuned = tuned_pipeline.predict(predict_input)
            preds_tuned = predicted_values_tuned.predict

            mse_value = mean_squared_error(y_data_test, preds_tuned,
                                           squared=False)
            mae_value = mean_absolute_error(y_data_test, preds_tuned)

            print(f'Obtained metrics for current iteration {i} after tuning:')
            print(f'RMSE - {mse_value:.2f}')
            print(f'MAE - {mae_value:.2f}\n')


if __name__ == '__main__':
    node_encoder = PrimaryNode('one_hot_encoding')
    node_scaling = SecondaryNode('scaling', nodes_from=[node_encoder])
    node_ridge = SecondaryNode('ridge', nodes_from=[node_scaling])
    node_lasso = SecondaryNode('lasso', nodes_from=[node_scaling])
    node_final = SecondaryNode('rfr', nodes_from=[node_ridge, node_lasso])

    init_pipeline = Pipeline(node_final)

    # Available tuners for application: PipelineTuner, NodesTuner
    run_river_experiment(file_path='../data/river_levels/station_levels.csv',
                         pipeline=init_pipeline,
                         iterations=20,
                         tuner=PipelineTuner)
