# Additional imports are required
import time

# Plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

rcParams['figure.figsize'] = 26, 11

import warnings

warnings.filterwarnings('ignore')

# Prerocessing for FEDOT
from fedot.core.repository.tasks import TsForecastingParams

from fedot.api.main import Fedot


def calculate_metrics(target, predicted):
    rmse = mean_squared_error(target, predicted, squared=True)
    mae = mean_absolute_error(target, predicted)
    mape = mean_absolute_percentage_error(target, predicted)
    return rmse, mae, mape


def plot_results(actual_time_series, predicted_values, len_train_data, y_name='Parameter'):
    """
    Function for drawing plot with predictions

    :param actual_time_series: the entire array with one-dimensional data
    :param predicted_values: array with predicted values
    :param len_train_data: number of elements in the training sample
    :param y_name: name of the y axis
    """

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
    plt.savefig('plt.jpg')
    plt.show()


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


data = pd.read_csv('./dataset/dataset_ML_pre.csv', sep=',')
data = reduce_memory_usage(data)
data.iloc[:, :].head()

# Specify forecast length
len_forecast = 55
name = 'Fedot_auto_ML'

# Got train, test parts, and the entire data
true_values = np.array(data['close_vlm'])
train_array = true_values[:-len_forecast]
test_array = true_values[-len_forecast:]
#
# plt.plot(true_values[26500:], label='Test')  # data['close_vlm'],
# # plt.plot( test_array, label = 'Test')
# plt.plot(train_array[26500:], label='Train')  # data['close_vlm'][:-len_forecast],
# plt.xlabel('Date', fontsize=15)
# plt.ylabel('Sea level, m', fontsize=15)
# plt.legend(fontsize=15)
# plt.grid()
plt.show()

# https://github.com/nccr-itmo/FEDOT/blob/master/cases/metocean_forecasting_problem.py

np.random.seed(144)
forecast_length = len_forecast

composer_params = {'max_depth': 7,
                   'max_arity': 5,
                   'num_of_generations': 155,
                   # 'timeout': 150,
                   # 'preset': 'ts',
                   'cv_folds': 3,
                   'validation_blocks': 3
                   }

len_feature = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
historical_data = data.iloc[24000:-len_forecast, len_feature].to_dict('list')
pd.DataFrame.from_dict(historical_data).to_csv('dataset.csv')

true_values = np.array(data['close_vlm'])
train_array = true_values[24000:-len_forecast]

timeout = -1
start = time.time()

fedot = Fedot(problem='ts_forecasting', timeout=timeout, seed=144, n_jobs=-1, composer_params=composer_params,
              task_params=TsForecastingParams(forecast_length=forecast_length))

pipeline = fedot.fit(features=historical_data, target=train_array)

end = time.time()

pipeline.show()
pipeline.show('pipeline_show.jpg')
pipeline.print_structure()

# Use model to obtain forecast
forecast = fedot.forecast(historical_data, forecast_length=forecast_length)

# Plot results
plot_results(actual_time_series=true_values[26700:],
             predicted_values=forecast,
             len_train_data=len(true_values[26700:]) - forecast_length)

# Print metrics for validation part
# test_array = train_array[-len_forecast:]
# metric = fedot.get_metrics(metric_names=['rmse', 'mae', 'mape'], target=test_array)
# print(metric)
rmse, mae, mape = calculate_metrics(test_array, forecast)
print(f'RMSE: {round(rmse, 3)}')
print(f'MAE: {round(mae, 3)}')
print(f'MAPE: {round(mape, 3)}')

rmse, mae, mape = calculate_metrics(test_array, forecast)
print(f'RMSE: {round(rmse, 3)}')
print(f'MAE: {round(mae, 3)}')
print(f'MAPE: {round(mape, 3)}')
