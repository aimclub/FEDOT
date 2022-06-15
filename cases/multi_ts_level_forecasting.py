import os
import numpy as np

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def prepare_data(forecast_length, multi_ts):
    """
    Function to form InputData from file with time-series
    """
    columns_to_use = ['61_91', '56_86', '61_86', '66_86']
    target_column = '61_91'
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))
    file_path = os.path.join(str(fedot_project_root()), 'cases/data/arctic/topaz_multi_ts.csv')
    if multi_ts:
        data = InputData.from_csv_multi_time_series(
            file_path=file_path,
            task=task,
            columns_to_use=columns_to_use)
    else:
        data = InputData.from_csv_time_series(
            file_path=file_path,
            task=task,
            target_column=target_column)
    train_data, test_data = train_test_data_setup(data)
    return train_data, test_data, task


def initial_pipeline():
    """
        Return pipeline with the following structure:
        lagged - ridge \
                        -> ridge -> final forecast
        lagged - ridge /
    """
    node_lagged_1 = PrimaryNode("lagged")
    node_lagged_1.custom_params = {'window_size': 50}

    node_smoth = PrimaryNode("smoothing")
    node_lagged_2 = SecondaryNode("lagged", nodes_from=[node_smoth])
    node_lagged_2.custom_params = {'window_size': 30}

    node_ridge = SecondaryNode("ridge", nodes_from=[node_lagged_1])
    node_lasso = SecondaryNode("lasso", nodes_from=[node_lagged_2])

    node_final = SecondaryNode("ridge", nodes_from=[node_ridge, node_lasso])
    pipeline = Pipeline(node_final)
    return pipeline


def visualize_result(train, test, target, forecast, multi_ts):
    if multi_ts:
        history = np.ravel(train.target[:, 0])
    else:
        history = np.ravel(train.target)
    plt.plot(np.ravel(test.idx), target, label='test')
    plt.plot(np.ravel(train.idx), history, label='history')
    plt.plot(np.ravel(test.idx), forecast, label='prediction_after_tuning')
    plt.xlabel('Time step')
    plt.ylabel('Sea level')
    plt.legend()
    plt.show()


def run_multi_ts_forecast(forecast_length, multi_ts):
    """
    Function for run experiment with use multi_ts data type (multi_ts=True) for train set extension
    Or run experiment on one time-series (multi_ts=False)
    """
    train_data, test_data, task = prepare_data(forecast_length, multi_ts)
    # init model for the time series forecasting
    init_pipeline = initial_pipeline()
    model = Fedot(problem='ts_forecasting',
                  task_params=task.task_params,
                  timeout=5,
                  initial_assumption=init_pipeline,
                  n_jobs=1,
                  composer_params={
                      'max_depth': 5,
                      'num_of_generations': 20,
                      'pop_size': 15,
                      'max_arity': 4,
                      'cv_folds': None,
                      'validation_blocks': None,
                      'available_operations': ['lagged', 'smoothing', 'diff_filter', 'gaussian_filter',
                                               'ridge', 'lasso', 'linear', 'cut']
                  })
    # fit model
    pipeline = model.fit(train_data)
    pipeline.show()

    # use model to obtain forecast
    forecast = model.predict(test_data)
    target = np.ravel(test_data.target)

    # visualize results
    visualize_result(train_data, test_data, target, forecast, multi_ts)

    print(f'MAE: {mean_absolute_error(target, forecast)}')
    print(f'RMSE: {mean_squared_error(target, forecast)}')
    print(f'MAPE: {mean_absolute_percentage_error(target, forecast)}')

    print(model.get_metrics(metric_names=['rmse', 'mae', 'mape'], target=target))


if __name__ == '__main__':
    forecast_length = 60
    run_multi_ts_forecast(forecast_length, multi_ts=True)
    run_multi_ts_forecast(forecast_length, multi_ts=False)
