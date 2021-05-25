import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.models.custom_model import CustomModelImplementation


def prepare_data(path_to_file, ts_name, exog_name, len_forecast=250):
    df = pd.read_csv(path_to_file)
    time_series = np.array(df[ts_name])
    exog_variable = np.array(df[exog_name])

    # Let's divide our data on train and test samples
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    # Exog feature
    train_data_exog = exog_variable[:-len_forecast]
    test_data_exog = exog_variable[-len_forecast:]

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    ###for main ts
    train_input = InputData(idx=np.arange(0, len(train_data)),
                            features=train_data,
                            target=train_data,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(train_data)
    end_forecast = start_forecast + len_forecast

    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=train_data,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    ###for exog ts
    train_input_exog = InputData(idx=np.arange(0, len(train_data_exog)),
                                 features=train_data_exog,
                                 target=train_data,
                                 task=task,
                                 data_type=DataTypesEnum.ts)

    predict_input_exog = InputData(idx=np.arange(start_forecast, end_forecast),
                                   features=test_data_exog,
                                   target=None,
                                   task=task,
                                   data_type=DataTypesEnum.ts)

    input_data = np.vstack([train_input, train_input_exog, predict_input, predict_input_exog])
    return input_data, test_data


if __name__ == '__main__':
    c_input, test_data = prepare_data(path_to_file='../notebooks/data/ts_sea_level.csv',
                           ts_name='Level',
                           exog_name='Neighboring level',
                           len_forecast=250)

    a = CustomModelImplementation()

    a.fit(c_input)
    pr = a.predict()

    predicted = np.ravel(np.array(pr))
    test_data = np.ravel(test_data)

    print(f'Predicted values: {predicted[:5]}')
    print(f'Actual values: {test_data[:5]}')

    mse_before = mean_squared_error(test_data, predicted, squared=False)
    mae_before = mean_absolute_error(test_data, predicted)
    print(f'RMSE - {mse_before:.4f}')
    print(f'MAE - {mae_before:.4f}\n')