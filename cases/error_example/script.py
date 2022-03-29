import os
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 29, 13

import matplotlib.pyplot as plt
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.api.main import Fedot


def reduce_memory_usage(df):
    start_memory = df.memory_usage().sum() / 1024 ** 2
    print(f"Memory usage of dataframe is {start_memory} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    pass
        else:
            df[col] = df[col].astype('category')

    end_memory = df.memory_usage().sum() / 1024 ** 2
    print(f"Memory usage of dataframe after reduction {end_memory} MB")
    print(f"Reduced by {100 * (start_memory - end_memory) / start_memory} % ")
    return df


def plot_results(actual_time_series, predicted_values, len_train_data, y_name='Parameter'):
    plt.plot(np.arange(0, len(actual_time_series)),
             actual_time_series, label='Actual values', c='green')
    plt.plot(np.arange(len_train_data, len_train_data + len(predicted_values)),
             predicted_values, label='Predicted', c='blue')
    # Plot black line which divide our array into train and test
    plt.plot([len_train_data, len_train_data],
             [min(actual_time_series), max(actual_time_series)], c='black', linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.show()


data = pd.read_csv('dataset_ML_pre_n.csv', sep=',', header=None)
data = reduce_memory_usage(data)

len_forecast = 34
name = 'Fedot_auto_ML_2'

np.random.seed(144)
forecast_length = len_forecast

composer_params = {'max_depth': 5,
                   'max_arity': 3,
                   'cv_folds': 5,
                   'validation_blocks': 5
                   }
len_feature = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
historical_data = data.iloc[21000:-len_forecast, len_feature].to_dict('list')
pd.DataFrame.from_dict(historical_data).to_csv('dataset.csv')
true_values = np.array(data.iloc[:, 0])
train_array = true_values[21000:-len_forecast]

timeout = 10
start = time.time()
fedot = Fedot(problem='ts_forecasting',
              timeout=timeout, seed=144,
              n_jobs=10,
              composer_params=composer_params,
              task_params=TsForecastingParams(forecast_length=forecast_length))
pipeline = fedot.fit(features=historical_data, target=train_array)
end = time.time()

pipeline.show('pipeline_show.jpg')

forecast = fedot.forecast(historical_data, forecast_length=forecast_length)

plot_results(actual_time_series=true_values[26500:], predicted_values=forecast,
             len_train_data=len(true_values[26500:])-forecast_length)

test_array = train_array[-len_forecast:]
metric = fedot.get_metrics(metric_names=['rmse', 'mae', 'mape'], target=test_array)
print(metric)