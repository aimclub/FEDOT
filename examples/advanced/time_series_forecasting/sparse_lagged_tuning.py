import os
import timeit
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from examples.simple.time_series_forecasting.ts_pipelines import ts_complex_dtreg_pipeline
from fedot.core.data.data import InputData
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root

warnings.filterwarnings('ignore')


def prepare_train_test_input(train_part, len_forecast):
    """ Function return prepared data for fit and predict

    :param len_forecast: forecast length
    :param train_part: time series which can be used as predictors for train

    :return train_input: Input Data for fit
    :return predict_input: Input Data for predict
    :return task: Time series forecasting task with parameters
    """

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_input = InputData(idx=np.arange(0, len(train_part)),
                            features=train_part,
                            target=train_part,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(train_part)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=train_part,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input, task


def run_tuning_test(pipeline, train_input, predict_input, test_data, task, show_metrics=True):
    """
    Function for predicting values in a time series

    :param pipeline: TsForecastingPipeline object
    :param train_input: InputData for fit
    :param predict_input: InputData for predict
    :param test_data: numpy array for validation
    :param task: Ts_forecasting task
    :param show_metrics: is there need to print metrics before and after tuning

    :return amount_of_seconds time spent on tuning
    :return mae_before MAE metric of pipeline without tuning
    :return mae_after MAE metric of pipeline with tuning
    """

    pipeline.fit_from_scratch(train_input)

    # Predict
    predicted_values = pipeline.predict(predict_input)
    old_predicted_values = predicted_values.predict
    cv_folds = 3
    validation_blocks = 2

    start_time = timeit.default_timer()
    pipeline_tuner = TunerBuilder(task)\
        .with_tuner(PipelineTuner)\
        .with_metric(RegressionMetricsEnum.MAE)\
        .with_cv_folds(cv_folds) \
        .with_validation_blocks(validation_blocks)\
        .with_iterations(20) \
        .build(train_input)
    pipeline = pipeline_tuner.tune(pipeline)
    print(pipeline.print_structure())
    amount_of_seconds = timeit.default_timer() - start_time
    print(f'\nTuning pipline on {amount_of_seconds:.2f} seconds\n')

    # Fit pipeline on the entire train data
    pipeline.fit_from_scratch(train_input)
    # Predict
    predicted_values = pipeline.predict(predict_input)
    new_predicted_values = predicted_values.predict
    new_predicted_values = np.ravel(np.array(new_predicted_values))
    old_predicted_values = np.ravel(np.array(old_predicted_values))

    mae_before = mean_absolute_error(test_data, old_predicted_values)
    mae_after = mean_absolute_error(test_data, new_predicted_values)

    if show_metrics:
        print(f'MAE before tuning - {mae_before:.4f}')
        print(f'MAE after tuning - {mae_after:.4f}\n')

    return amount_of_seconds, mae_before, mae_after


def visualize(tuned, no_tuned, time, method_name):
    ind = np.arange(len(tuned))
    width = 0.4

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.bar(ind, no_tuned, width, fc=(0, 0, 1, 0.5), label='No tuning')
    ax1.bar(ind + width, tuned, width, fc=(1, 0, 0, 0.5), label='Tuned')
    ax1.set_ylabel('MAE', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.plot(ind + width / 2, time, color=color)
    ax2.set_ylabel('time, s', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'MAE with tuning process based on {method_name}')
    fig.legend()
    fig.tight_layout()
    plt.show()


def run_tuning_comparison(n_repits=10, ts_size=1000, forecast_length=50, visualization=True):
    file_path = os.path.join(str(fedot_project_root()), 'cases/data/time_series/temperature.csv')
    df = pd.read_csv(file_path)
    time_series = np.array(df['value'])[:ts_size]

    # Train/test split
    train_part = time_series[:-forecast_length]
    test_part = time_series[-forecast_length:]

    # Prepare data for train and test
    train_input, predict_input, task = prepare_train_test_input(train_part, forecast_length)

    nodes_names = ['sparse_lagged', 'lagged']
    for name in nodes_names:

        # set lists for data collecting
        time_list = []
        mae_no_tuning = []
        mae_tuning = []

        test_part = np.ravel(test_part)
        # tuning calculations for averaging
        for i in range(n_repits):
            pipeline = ts_complex_dtreg_pipeline(name)
            amount_of_seconds, mae_before, mae_after = run_tuning_test(pipeline,
                                                                       train_input,
                                                                       predict_input,
                                                                       test_part,
                                                                       task)
            time_list.append(amount_of_seconds)
            mae_no_tuning.append(mae_before)
            mae_tuning.append(mae_after)

        if visualization:
            visualize(mae_tuning, mae_no_tuning, time_list, name)

        print(f'Mean time: {np.array(time_list).mean()}')
        print(f'Mean MAE without tuning: {np.array(mae_no_tuning).mean()}')
        print(f'Mean MAE with tuning: {np.array(mae_tuning).mean()}')


if __name__ == '__main__':
    # On large time series the speed of sparse_lagged increase (ts_size parameter)
    run_tuning_comparison(n_repits=10, ts_size=1000, forecast_length=50, visualization=True)
