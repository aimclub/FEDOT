import timeit
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.pipelines.tuning.unified import PipelineTuner
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')


def get_pipeline(first_node='lagged'):
    """
        Return pipeline with the following structure:
        lagged/sparse_lagged - ridge
    """

    node_lagged = PrimaryNode(first_node)
    node_final = SecondaryNode('dtreg', nodes_from=[node_lagged])
    pipeline = Pipeline(node_final)

    return pipeline


def prepare_train_test_input(train_part, len_forecast):
    """ Function return prepared data for fit and predict

    :param len_forecast: forecast length
    :param train_part: time series which can be used as predictors for train

    :return train_input: Input Data for fit
    :return predict_input: Input Data for predict
    :return task: Time series forecasting task with parameters
    """

    # Specify the task to solve
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


def run_tuning_test(pipeline, train_input, predict_input, test_data, task):
    """
    Function for predicting values in a time series

    :param pipeline: TsForecastingPipeline object
    :param train_input: InputData for fit
    :param predict_input: InputData for predict
    :param task: Ts_forecasting task

    :return predicted_values: numpy array, forecast of model
    """

    pipeline.fit_from_scratch(train_input)

    # Predict
    predicted_values = pipeline.predict(predict_input)
    old_predicted_values = predicted_values.predict
    old_predicted_values = np.ravel(np.array(old_predicted_values))

    start_time = timeit.default_timer()
    pipeline_tuner = PipelineTuner(pipeline=pipeline, task=task,
                                   iterations=10)
    pipeline = pipeline_tuner.tune_pipeline(input_data=train_input,
                                            loss_function=mean_absolute_error,
                                            cv_folds=3,
                                            validation_blocks=2)
    print(pipeline.print_structure())
    amount_of_seconds = timeit.default_timer() - start_time
    print(f'\nTuning pipline on {amount_of_seconds:.2f} seconds\n')

    # Fit pipeline on the entire train data
    pipeline.fit_from_scratch(train_input)
    # Predict
    predicted_values = pipeline.predict(predict_input)
    new_predicted_values = predicted_values.predict
    new_predicted_values = np.ravel(np.array(new_predicted_values))

    mse_before = mean_squared_error(test_data, old_predicted_values, squared=False)
    mae_before = mean_absolute_error(test_data, old_predicted_values)
    print(f'RMSE before tuning - {mse_before:.4f}')
    print(f'MAE before tuning - {mae_before:.4f}\n')

    mse_after = mean_squared_error(test_data, new_predicted_values, squared=False)
    mae_after = mean_absolute_error(test_data, new_predicted_values)
    print(f'RMSE after tuning - {mse_after:.4f}')
    print(f'MAE after tuning - {mae_after:.4f}\n')

    return amount_of_seconds, mse_before, mse_after, mae_before, mae_after


def run_ts_forecasting_problem(forecast_length=50):
    file_path = '../cases/data/time_series/temperature.csv'

    df = pd.read_csv(file_path)
    time_series = np.array(df['value'])[-1000:]

    # Train/test split
    train_part = time_series[:-forecast_length]
    test_part = time_series[-forecast_length:]

    # Prepare data for train and test
    train_input, predict_input, task = prepare_train_test_input(train_part, forecast_length)

    pipeline = get_pipeline('sparse_lagged')
    #pipeline = get_pipeline('lagged')

    time_list = []
    mse_no_tuning = []
    mse_tuning = []
    mae_no_tuning = []
    mae_tuning = []

    test_part = np.ravel(test_part)
    for i in range(10):
        amount_of_seconds, mse_before, mse_after, mae_before, mae_after = run_tuning_test(pipeline,
                                                                                          train_input,
                                                                                          predict_input,
                                                                                          test_part,
                                                                                          task)
        time_list.append(amount_of_seconds)
        mse_no_tuning.append(mse_before)
        mse_tuning.append(mse_after)
        mae_no_tuning.append(mae_before)
        mae_tuning.append(mae_after)

    print('Time for tuning:')
    print(time_list)
    print('MSE without tuning:')
    print(mse_no_tuning)
    print('MSE with tuning:')
    print(mse_tuning)
    print('MAE without tuning:')
    print(mae_no_tuning)
    print('MAE with tuning:')
    print(mae_tuning)

    plt.bar(np.arange(len(mse_no_tuning)), mse_no_tuning, fc=(0, 0, 1, 0.5), label='No tuning')
    plt.bar(np.arange(len(mse_tuning)), mse_tuning, fc=(1, 0, 0, 0.5), label='Tuned')
    plt.legend()
    plt.title('MSE')
    plt.show()

    plt.bar(np.arange(len(mae_no_tuning)), mae_no_tuning, fc=(0, 0, 1, 0.5), label='No tuning')
    plt.bar(np.arange(len(mae_tuning)), mae_tuning, fc=(1, 0, 0, 0.5), label='Tuned')
    plt.legend()
    plt.title('MAE')
    plt.show()

    print(f'Mean time: {np.array(time_list).mean()}')

if __name__ == '__main__':
    run_ts_forecasting_problem()
