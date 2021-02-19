import os
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from pylab import rcParams
rcParams['figure.figsize'] = 12, 6
from scipy import interpolate
import scipy


def plot_heat(df, len_col, window_col, target_col, cmap='jet', improve_neg=False,
              int_method='nearest'):
    """
    Функция для отрисовки heat map
    :param df: датафрейм для процессинга
    :param len_col: длина прогноза
    :param window_col: размер скользящего окна
    :param target_col: отклик (или интересующая колонка)
    """

    len_arr = np.array(df[len_col].unique())
    window_arr = np.array(df[window_col].unique())

    # Пустая матрица
    empty_matrix = np.zeros((max(len_arr)+1, max(window_arr)+1))

    for i in len_arr:
        df_i = df[df[len_col] == i]
        for j in window_arr:
            try:
                df_ij = df_i[df_i[window_col] == j]
                target_value = np.array(df_ij[target_col])
                target_value = float(target_value[0])
                empty_matrix[i][j] = target_value
            except Exception:
                pass

    # Интерполяция
    masked_array = np.ma.masked_where(empty_matrix == 0.0, empty_matrix)
    x = np.arange(0, len(empty_matrix[0]))
    y = np.arange(0, len(empty_matrix))
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~masked_array.mask]
    y1 = yy[~masked_array.mask]
    new_arr = masked_array[~masked_array.mask]

    if int_method == 'nearest':
        int_matrix = interpolate.griddata((x1, y1), new_arr.ravel(), (xx, yy), method='nearest')

        # Удаление артефактов после интерполяции, если это требуется
        if improve_neg:
            # Границы, в которых осуществлялся поиск - от 1 до 4х
            int_matrix[int_matrix < 1.0] = 1.0
            int_matrix[int_matrix > 4.0] = 4.0
            int_matrix = int_matrix.round(0)

        # Отрисовка матрицы
        cmap = cm.get_cmap(cmap)
        plt.imshow(int_matrix, interpolation='nearest', cmap=cmap)
        plt.colorbar(label=target_col)
        plt.ylabel(len_col, fontsize=15)
        plt.xlabel(window_col, fontsize=15)
        for i in len_arr:
            for j in window_arr:
                plt.scatter(j, i, c='black', s=2)
        plt.show()

    else:
        int_matrix = interpolate.griddata((x1, y1), new_arr.ravel(), (xx, yy),
                                          method='cubic')
        # Части, которые не были проинтерполированны - дозаполняются
        # интерполяцией методом ближайшего соседа
        ids = np.argwhere(empty_matrix == 0.0)
        for id in ids:
            interpolated_value = int_matrix[id[0], id[1]]
            if np.isnan(interpolated_value):
                pass
            else:
                empty_matrix[id[0], id[1]] = interpolated_value
        masked_array = np.ma.masked_where(empty_matrix == 0.0, empty_matrix)
        x = np.arange(0, len(empty_matrix[0]))
        y = np.arange(0, len(empty_matrix))
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~masked_array.mask]
        y1 = yy[~masked_array.mask]
        new_arr = masked_array[~masked_array.mask]
        int_nearest = interpolate.griddata((x1, y1), new_arr.ravel(), (xx, yy),
                                           method='nearest')
        ids = np.argwhere(np.isnan(int_matrix))
        for id in ids:
            int_matrix[id[0], id[1]] = int_nearest[id[0], id[1]]

        # Удаление артефактов после интерполяции, если это требуется
        if improve_neg:
            # Границы, в которых осуществлялся поиск - от 1 до 4х
            int_matrix[int_matrix < 1.0] = 1.0
            int_matrix[int_matrix > 4.0] = 4.0
            int_matrix = int_matrix.round(0)

        # Отрисовка матрицы
        cmap = cm.get_cmap(cmap)
        plt.imshow(int_matrix, interpolation='bicubic', cmap=cmap)
        plt.colorbar(label=target_col)
        plt.ylabel(len_col, fontsize=15)
        plt.xlabel(window_col, fontsize=15)
        for i in len_arr:
            for j in window_arr:
                plt.scatter(j, i, c='black', s=2)
        plt.show()


def create_dataframe(folder, time_series='mean'):
    """ Функция создает датафрейм с заданными параметрами для визуализации

    :param folder: папка, в которой хранятся все необходимые файлы csv
    :param time_series: для какого временного ряда требуется формировать
    датафрейм, если 'mean' - то берется среднее значение по всем
    """

    files = os.listdir(folder)
    files.sort()

    arrays_df = []
    reports = []
    for file in files:
        if file.endswith("report_.csv"):
            reports.append(file)
        else:
            arrays_df.append(file)

    for index, report_file in enumerate(reports):
        print(f'Processing {report_file}')
        report_df = pd.read_csv(os.path.join(folder, report_file))

        if time_series != 'mean':
            time_series_df = report_df[report_df['Time series label'] == time_series]
        else:
            time_series_df = report_df.groupby(['Size']).agg({'MAE': 'mean',
                                                              'MAPE': 'mean',
                                                              'Time': 'mean',
                                                              'Depth': 'mean'})
            time_series_df = time_series_df.reset_index()

        splitted = report_file.split('_')
        forecast_len = int(splitted[0])
        result_df = time_series_df[['Size', 'MAPE', 'Time', 'Depth']]
        result_df['Forecast len'] = [forecast_len]*len(result_df)

        if index == 0:
            response_df = result_df
        else:
            frames = [response_df, result_df]
            response_df = pd.concat(frames)

    return response_df

"""
'GS10' 
'EXCHUS' 
'EXCAUS'
'Weekly U.S. Refiner and Blender Adjusted Net Production of Finished Motor Gasoline  (Thousand Barrels per Day)'
'Weekly Minnesota Midgrade Conventional Retail Gasoline Prices  (Dollars per Gallon)'
'Weekly U.S. Percent Utilization of Refinery Operable Capacity (Percent)'
'Weekly U.S. Exports of Crude Oil and Petroleum Products  (Thousand Barrels per Day)'
'Weekly U.S. Field Production of Crude Oil  (Thousand Barrels per Day)'
'Weekly U.S. Ending Stocks of Crude Oil and Petroleum Products  (Thousand Barrels)'
'Weekly U.S. Product Supplied of Finished Motor Gasoline  (Thousand Barrels per Day)'
"""

path = 'D:/time_series_exp/fedot_exp'
dataframe = create_dataframe(folder=path,
                             time_series='mean')


plot_heat(dataframe, len_col='Forecast len', window_col='Size',
          target_col='Depth', cmap='coolwarm', improve_neg=True,
          int_method='cubic')

plot_heat(dataframe, len_col='Forecast len', window_col='Size',
          target_col='MAPE', cmap='coolwarm', improve_neg=False,
          int_method='cubic')


print(f'Mean MAPE value - {dataframe["MAPE"].mean():.2f}')
print(f'Mean time - {dataframe["Time"].mean():.2f}')
print(f'Mean depth - {dataframe["Depth"].mean():.2f}')





