from datetime import timedelta

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from examples.simple.regression.regression_pipelines import regression_ransac_pipeline
from fedot.core.data.data import InputData
from fedot.core.pipelines.tuning.sequential import SequentialTuner
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.synth_dataset_generator import regression_dataset

np.random.seed(2020)


def get_regression_dataset(features_options, samples_amount=250,
                           features_amount=5):
    """
    Prepares four numpy arrays with different scale features and target
    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param features_options: The dictionary containing features options in key-value
    format:
        - informative: the amount of informative features;
        - bias: bias term in the underlying linear model;
    :return x_data_train: features to train
    :return y_data_train: target to train
    :return x_data_test: features to test
    :return y_data_test: target to test
    """

    x_data, y_data = regression_dataset(samples_amount=samples_amount,
                                        features_amount=features_amount,
                                        features_options=features_options,
                                        n_targets=1,
                                        noise=0.0, shuffle=True)

    # Changing the scale of the data
    for i, coeff in zip(range(0, features_amount),
                        np.random.randint(1, 100, features_amount)):
        # Get column
        feature = np.array(x_data[:, i])

        # Change scale for this feature
        rescaled = feature * coeff
        x_data[:, i] = rescaled

    # Train and test split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        test_size=0.3)

    return x_train, y_train, x_test, y_test


def run_experiment(pipeline, tuner):
    samples = [50, 250, 150]
    features = [1, 5, 10]
    options = [{'informative': 1, 'bias': 0.0},
               {'informative': 2, 'bias': 2.0},
               {'informative': 1, 'bias': 3.0}]

    for samples_amount, features_amount, features_options in zip(samples, features, options):
        print('=======================================')
        print(f'\nAmount of samples {samples_amount}, '
              f'amount of features {features_amount}, '
              f'additional options {features_options}')

        x_train, y_train, x_test, y_test = get_regression_dataset(features_options,
                                                                  samples_amount,
                                                                  features_amount)

        # Define regression task
        task = Task(TaskTypesEnum.regression)

        # Prepare data to train the model
        train_input = InputData(idx=np.arange(0, len(x_train)),
                                features=x_train,
                                target=y_train,
                                task=task,
                                data_type=DataTypesEnum.table)

        predict_input = InputData(idx=np.arange(0, len(x_test)),
                                  features=x_test,
                                  target=None,
                                  task=task,
                                  data_type=DataTypesEnum.table)

        # Fit it
        pipeline.fit(train_input)

        # Predict
        predicted_values = pipeline.predict(predict_input)
        pipeline_prediction = predicted_values.predict

        mae_value = mean_absolute_error(y_test, pipeline_prediction)
        print(f'Mean absolute error - {mae_value:.4f}\n')

        if tuner is not None:
            print('Start tuning process ...')
            pipeline_tuner = TunerBuilder(task)\
                .with_tuner(tuner)\
                .with_metric(RegressionMetricsEnum.MAE)\
                .with_iterations(50) \
                .with_timeout(timedelta(seconds=50))\
                .build(train_input)
            tuned_pipeline = pipeline_tuner.tune(pipeline)

            # Fit it
            tuned_pipeline.fit(train_input)

            # Predict
            predicted_values_tuned = tuned_pipeline.predict(predict_input)
            preds_tuned = predicted_values_tuned.predict

            mae_value = mean_absolute_error(y_test, preds_tuned)

            print('Obtained metrics after tuning:')
            print(f'MAE - {mae_value:.4f}\n')

        pipeline.unfit()


# Script for testing is pipeline can process different datasets for regression task
if __name__ == '__main__':
    run_experiment(regression_ransac_pipeline(), tuner=SequentialTuner)
