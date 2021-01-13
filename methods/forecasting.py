import os
import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from matplotlib import pyplot as plt
import timeit
from pylab import rcParams
rcParams['figure.figsize'] = 18, 7
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import statsmodels.api as sm
import pylab

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.ts_chain import TsForecastingChain
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


# Расчет метрики - cредняя абсолютная процентная ошибка
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    # У представленной ниже формулы есть недостаток, - если в массиве y_true есть хотя бы одно значение 0.0,
    # то по формуле np.mean(np.abs((y_true - y_pred) / y_true)) * 100 мы получаем inf, поэтому
    zero_indexes = np.argwhere(y_true == 0.0)
    for index in zero_indexes:
        y_true[index] = 0.001
    value = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return(value)

def parse_gap_ids(gap_list: list) -> list:
    """
    Method allows parsing source array with gaps indexes

    :param gap_list: array with indexes of gaps in array
    :return: a list with separated gaps in continuous intervals
    """

    new_gap_list = []
    local_gaps = []
    for index, gap in enumerate(gap_list):
        if index == 0:
            local_gaps.append(gap)
        else:
            prev_gap = gap_list[index - 1]
            if gap - prev_gap > 1:
                # There is a "gap" between gaps
                new_gap_list.append(local_gaps)

                local_gaps = []
                local_gaps.append(gap)
            else:
                local_gaps.append(gap)
    new_gap_list.append(local_gaps)

    return new_gap_list


# Функция для предсказания и подсчета статистик при прогнозе временного ряда
def forecasting_accuracy(path, prediction_len, vis = True):

    mapes_per_model = []
    models = []
    files = []

    for file_name in ['Synthetic.csv', 'Sea_hour.csv', 'Sea_10_240.csv']:
        # Исходный файл с пропусками
        gap_path = os.path.join(path, file_name)
        gap_df = pd.read_csv(gap_path)
        gap_df['Date'] = pd.to_datetime(gap_df['Date'])

        # Простые методы
        linear_path = os.path.join(os.path.join(path, 'linear'), file_name)
        linear_df = pd.read_csv(linear_path)
        local_poly_path = os.path.join(os.path.join(path, 'poly'), file_name)
        local_poly_df = pd.read_csv(local_poly_path)
        batch_poly_path = os.path.join(os.path.join(path, 'batch_poly'), file_name)
        batch_poly_df = pd.read_csv(batch_poly_path)

        # Методы восстановления пропусков средствами языка R
        kalman_path = os.path.join(os.path.join(path, 'kalman'), file_name)
        kalman_df = pd.read_csv(kalman_path)
        ma_path = os.path.join(os.path.join(path, 'ma'), file_name)
        ma_df = pd.read_csv(ma_path)
        spline_path = os.path.join(os.path.join(path, 'spline'), file_name)
        spline_df = pd.read_csv(spline_path)

        # Методы восстановления пропусков FEDOT
        fedot_ridge_30_path = os.path.join(os.path.join(path, 'fedot_ridge_30'), file_name)
        fedot_ridge_30_df = pd.read_csv(fedot_ridge_30_path)
        fedot_ridge_100_path = os.path.join(os.path.join(path, 'fedot_ridge_100'), file_name)
        fedot_ridge_100_df = pd.read_csv(fedot_ridge_100_path)
        fedot_compose = os.path.join(os.path.join(path, 'fedot_composing'), file_name)
        fedot_compose_df = pd.read_csv(fedot_compose)

        # Исходный временной ряд без пропусков
        arr_parameter = np.array(gap_df['Height'])
        # Временной ряд с пропусками
        arr_mask = np.array(gap_df['gap'])
        ids_gaps = np.ravel(np.argwhere(arr_mask == -100.0))

        array_gaps = np.ma.masked_where(arr_mask == -100.0, arr_mask)

        # plt.plot(gap_df['Date'], gap_df['Height'], c='red', alpha=0.5)
        # plt.ylabel('Sea level, m', fontsize=15)
        # plt.xlabel('Date', fontsize=15)
        # plt.grid()
        # plt.show()

        # autocorrelation
        # sm.graphics.tsa.plot_acf(gap_df['Height'],lags=250)
        # pylab.show()
        # print("Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(gap_df['Height'])[1])

        if vis:
            plt.plot(gap_df['Date'], arr_parameter, c='red', alpha=0.2)
            for index in ids_gaps:
                plt.plot([gap_df['Date'][index], gap_df['Date'][index]], [min(arr_parameter), arr_parameter[index]], c = 'red', alpha=0.05)
            plt.plot(gap_df['Date'], array_gaps, c='blue', alpha=1.0)
            plt.ylabel('Sea level, m', fontsize=15)
            plt.xlabel('Date', fontsize=15)
            plt.grid()
            plt.show()

        withoutgap_arr_linear = np.array(linear_df['gap'])
        withoutgap_arr_local = np.array(local_poly_df['gap'])
        withoutgap_arr_batch = np.array(batch_poly_df['gap'])

        withoutgap_arr_kalman = np.array(kalman_df['gap'])
        withoutgap_arr_ma = np.array(ma_df['gap'])
        withoutgap_arr_spline = np.array(spline_df['gap'])

        withoutgap_arr_ridge_30 = np.array(fedot_ridge_30_df['gap'])
        withoutgap_arr_ridge_100 = np.array(fedot_ridge_100_df['gap'])
        withoutgap_arr_compose = np.array(fedot_compose_df['gap'])


        if vis:
            plt.plot(gap_df['Date'], arr_parameter, c='green', alpha=0.5,
                 label='Actual values')
            plt.plot(gap_df['Date'], withoutgap_arr_linear, c='red', alpha=0.5,
                     label='Linear interpolation')
            plt.plot(gap_df['Date'], withoutgap_arr_local, c='orange', alpha=0.5,
                     label='Local polynomial approximation')
            plt.plot(gap_df['Date'], withoutgap_arr_batch, c='purple', alpha=0.5,
                     label='Batch polynomial approximation')
            plt.plot(gap_df['Date'], array_gaps, c='blue', alpha=1.0)
            plt.ylabel('Sea level, m', fontsize=15)
            plt.xlabel('Date', fontsize=15)
            plt.grid()
            plt.legend(fontsize=15)
            plt.show()

            plt.plot(gap_df['Date'], arr_parameter, c='green', alpha=0.5,
                     label='Actual values')
            plt.plot(gap_df['Date'], withoutgap_arr_kalman, c='red', alpha=0.5,
                     label='Kalman filtering')
            plt.plot(gap_df['Date'], withoutgap_arr_ma, c='orange', alpha=0.5,
                     label='Moving average')
            plt.plot(gap_df['Date'], withoutgap_arr_spline, c='purple', alpha=0.5,
                     label='Spline interpolation')
            plt.plot(gap_df['Date'], array_gaps, c='blue', alpha=1.0)
            plt.ylabel('Sea level, m', fontsize=15)
            plt.xlabel('Date', fontsize=15)
            plt.grid()
            plt.legend(fontsize=15)
            plt.show()

            plt.plot(gap_df['Date'], arr_parameter, c='green', alpha=0.5,
                     label='Actual values')
            plt.plot(gap_df['Date'], withoutgap_arr_batch, c='red',
                     alpha=0.5,
                     label='Batch polynomial approximation')
            plt.plot(gap_df['Date'], withoutgap_arr_kalman, c='orange', alpha=0.5,
                     label='Kalman filtering')
            plt.plot(gap_df['Date'], withoutgap_arr_ridge_30, c='purple', alpha=0.5,
                     label='Ridge 30 ws')
            plt.plot(gap_df['Date'], array_gaps, c='blue', alpha=1.0)
            plt.ylabel('Sea level, m', fontsize=15)
            plt.xlabel('Date', fontsize=15)
            plt.grid()
            plt.legend(fontsize=15)
            plt.show()

        train_part = arr_parameter[:-prediction_len]
        test_part = arr_parameter[-prediction_len:]

        # Подготавливаем часть временного ряда с восстановленными значениями
        train_part_linear = withoutgap_arr_linear[:-prediction_len]
        train_part_local = withoutgap_arr_local[:-prediction_len]
        train_part_batch = withoutgap_arr_batch[:-prediction_len]

        train_part_kalman = withoutgap_arr_kalman[:-prediction_len]
        train_part_ma = withoutgap_arr_ma[:-prediction_len]
        train_part_stine = withoutgap_arr_spline[:-prediction_len]

        train_part_ridge_30 = withoutgap_arr_ridge_30[:-prediction_len]
        train_part_ridge_100 = withoutgap_arr_ridge_100[:-prediction_len]
        train_part_compose = withoutgap_arr_compose[:-prediction_len]

        if file_name == 'Hour_data_m.csv':
            max_window_size = 50
        else:
            max_window_size = 500
        for sample, model in zip([train_part, train_part_linear, train_part_local, train_part_batch,
                       train_part_kalman, train_part_ma, train_part_stine, train_part_ridge_30,
                       train_part_ridge_100, train_part_compose], ['Original', 'Linear interpolation', 'Local polynomial approximation',
                                               'Batch polynomial approximation', 'Kalman filtering', 'Moving average',
                                               'Spline interpolation', 'Ridge forward 30 ws', 'Ridge forward 100 ws', 'Chain compose']):
            node_first = PrimaryNode('ridge')
            node_second = PrimaryNode('ridge')
            node_trend_model = SecondaryNode('linear', nodes_from=[node_first])
            node_residual_model = SecondaryNode('linear', nodes_from=[node_second])

            node_final = SecondaryNode('svr', nodes_from=[node_trend_model,
                                                            node_residual_model])
            chain = TsForecastingChain(node_final)

            task = Task(TaskTypesEnum.ts_forecasting,
                        TsForecastingParams(forecast_length=prediction_len,
                                            max_window_size=max_window_size,
                                            return_all_steps=False,
                                            make_future_prediction=True))

            input_data = InputData(idx=np.arange(0, len(sample)),
                                   features=None,
                                   target=sample,
                                   task=task,
                                   data_type=DataTypesEnum.ts)

            chain.fit_from_scratch(input_data)

            # "Test data" for making prediction for a specific length
            test_data = InputData(idx=np.arange(0, prediction_len),
                                  features=None,
                                  target=None,
                                  task=task,
                                  data_type=DataTypesEnum.ts)

            predicted_values = chain.forecast(initial_data=input_data,
                                              supplementary_data=test_data).predict

            print(model)
            MAE = mean_absolute_error(test_part, predicted_values)
            print('Mean absolute error -', round(MAE, 4))

            RMSE = (mean_squared_error(test_part, predicted_values)) ** 0.5
            print('RMSE -', round(RMSE, 4))

            MedianAE = median_absolute_error(test_part, predicted_values)
            print('Median absolute error -', round(MedianAE, 4))

            mape = mean_absolute_percentage_error(test_part, predicted_values)
            print('MAPE -', round(mape, 4), '\n')

            if file_name == 'Sea_10_240.csv':
                plt.plot(gap_df['Date'], arr_parameter, c = 'green', alpha = 0.5, label = 'Actual values')
                plt.plot(gap_df['Date'][:-prediction_len], sample, c = 'blue', label = 'Restored series')
                plt.plot(gap_df['Date'][-prediction_len:], predicted_values, c = 'red', alpha = 0.5, label = 'Model forecast')
                plt.ylabel('Sea level, m', fontsize=15)
                plt.xlabel('Date', fontsize=15)
                plt.grid()
                plt.title(model, fontsize=15)
                plt.legend(fontsize=15)
                plt.show()

            models.append(model)
            mapes_per_model.append(mape)
            files.append(file_name)

    local_df = pd.DataFrame({'MAPE': mapes_per_model,
                             'Model': models,
                             'File': files})

    for model in local_df['Model'].unique():
        local_local_df = local_df[local_df['Model'] == model]
        mape_arr = np.array(local_local_df['MAPE'])

        print(f'Среднее значение ошибки для модели {model} - {np.mean(mape_arr)}')
        for file in local_local_df['File'].unique():
            l_local_local_df = local_local_df[local_local_df['File']==file]
            print(f'{model}, {file}, MAPE - {float(l_local_local_df["MAPE"])}')

forecasting_accuracy(path = 'D:/iccs_article', prediction_len = 400, vis = False)
