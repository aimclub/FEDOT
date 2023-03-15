import datetime
import warnings

import numpy as np
import pandas as pd
from golem.core.tuning.simultaneous import SimultaneousTuner

from sklearn.metrics import mean_absolute_error, mean_squared_error

from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters

from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.quality_metrics_repository import \
    RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

warnings.filterwarnings('ignore')


def get_pipeline_info(pipeline):
    """ Function print info about pipeline and return operations in it and depth

    :param pipeline: pipeline to process
    :return obtained_operations: operations in the nodes
    :return depth: depth of the pipeline
    """

    obtained_operations = [str(node) for node in pipeline.nodes]
    depth = int(pipeline.depth)
    pipeline.print_structure()

    return obtained_operations, depth


def fit_predict_for_pipeline(pipeline, train_input, predict_input):
    """ Function apply fit and predict operations

    :param pipeline: pipeline to process
    :param train_input: InputData for fit
    :param predict_input: InputData for predict

    :return preds: prediction of the pipeline
    """
    # Fit it
    pipeline.fit_from_scratch(train_input)

    # Predict
    predicted_values = pipeline.predict(predict_input)
    preds = predicted_values.predict

    return preds


def run_river_composer_experiment(file_path, init_pipeline, file_to_save,
                                  iterations=20, tuner=None):
    """ Function launch experiment for river level prediction. Composing and
    tuner processes are available for such experiment.

    :param file_path: path to the file with river level data
    :param init_pipeline: pipeline to start composing process
    :param file_to_save: path to the file and file name to save report
    :param iterations: amount of iterations to process
    :param tuner: if tuning after composing process is required or not. tuner -
    SequentialTuner or SimultaneousTuner.
    """

    # Read dataframe and prepare train and test data
    data = InputData.from_csv(file_path, target_columns='level_station_2',
                              task=Task(TaskTypesEnum.regression),
                              columns_to_drop=['date'])
    train_input, predict_input = train_test_data_setup(data)
    y_data_test = np.array(predict_input.target)

    available_secondary_operations = ['ridge', 'lasso', 'dtreg',
                                      'rfr', 'adareg', 'knnreg',
                                      'linear', 'svr', 'poly_features',
                                      'scaling', 'ransac_lin_reg', 'rfe_lin_reg',
                                      'pca', 'ransac_non_lin_reg',
                                      'rfe_non_lin_reg', 'normalization']
    available_primary_operations = ['one_hot_encoding']
    metric_function = RegressionMetricsEnum.MAE
    # Report arrays
    obtained_pipelines = []
    depths = []
    maes = []
    for i in range(0, iterations):
        print(f'Iteration {i}\n')

        composer_requirements = PipelineComposerRequirements(
            primary=available_primary_operations,
            secondary=available_secondary_operations,
            max_arity=3, max_depth=8,
            num_of_generations=5,
            timeout=datetime.timedelta(minutes=5)
        )

        optimizer_parameters = GPAlgorithmParameters(
            pop_size=10, mutation_prob=0.8, crossover_prob=0.8,
        )

        composer = ComposerBuilder(task=data.task). \
            with_requirements(composer_requirements). \
            with_optimizer_params(optimizer_parameters). \
            with_metrics(metric_function). \
            with_initial_pipelines([init_pipeline]). \
            build()

        obtained_pipeline = composer.compose_pipeline(data=train_input)

        # Display info about obtained pipeline
        obtained_models, depth = get_pipeline_info(pipeline=obtained_pipeline)

        preds = fit_predict_for_pipeline(pipeline=obtained_pipeline,
                                         train_input=train_input,
                                         predict_input=predict_input)

        mse_value = mean_squared_error(y_data_test, preds, squared=False)
        mae_value = mean_absolute_error(y_data_test, preds)
        print(f'Obtained metrics for current iteration {i}:')
        print(f'RMSE - {mse_value:.2f}')
        print(f'MAE - {mae_value:.2f}\n')

        if tuner is not None:
            print('Start tuning process ...')
            pipeline_tuner = TunerBuilder(data.task)\
                .with_tuner(tuner)\
                .with_metric(metric_function)\
                .with_iterations(100)\
                .build(train_input)
            tuned_pipeline = pipeline_tuner.tune(obtained_pipeline)

            preds_tuned = fit_predict_for_pipeline(pipeline=tuned_pipeline,
                                                   train_input=train_input,
                                                   predict_input=predict_input)

            mse_value = mean_squared_error(y_data_test, preds_tuned, squared=False)
            mae_value = mean_absolute_error(y_data_test, preds_tuned)

            print(f'Obtained metrics for current iteration {i} after tuning:')
            print(f'RMSE - {mse_value:.2f}')
            print(f'MAE - {mae_value:.2f}\n')

        obtained_pipelines.append(obtained_models)
        maes.append(mae_value)
        depths.append(depth)

    report = pd.DataFrame({'Pipeline': obtained_pipelines,
                           'Depth': depths,
                           'MAE': maes})
    report.to_csv(file_to_save, index=False)


if __name__ == '__main__':
    # Define pipeline to start composing with it
    node_encoder = PipelineNode('one_hot_encoding')
    node_rans = PipelineNode('ransac_lin_reg', nodes_from=[node_encoder])
    node_scaling = PipelineNode('scaling', nodes_from=[node_rans])
    node_final = PipelineNode('linear', nodes_from=[node_scaling])

    init_pipeline = Pipeline(node_final)

    # Available tuners for application: SimultaneousTuner, SequentialTuner
    run_river_composer_experiment(file_path='../data/river_levels/station_levels.csv',
                                  init_pipeline=init_pipeline,
                                  file_to_save='../data/river_levels/old_composer_new_preprocessing_report.csv',
                                  iterations=20,
                                  tuner=SimultaneousTuner)
