import os
import pandas as pd
import numpy as np
import timeit
from scipy import interpolate
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from matplotlib import pyplot as plt
from datetime import datetime
from pylab import rcParams
rcParams['figure.figsize'] = 18, 7


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

# Функция восстановления временного ряда
# На основе https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
### Input:
# arr (np array)      --- одномерный массив (временной ряд) с пропусками, пропуски помечены bad_value
# gap_value (float)   --- флаг наличия пропуска
### Output:
# y (array) --- массив (временной ряд) без пропусков
def linear_interpolation(arr, gap_value = -100.0):
    # Индексы известных элементов
    non_nan = np.ravel(np.argwhere(arr != gap_value))
    # Все известные элементы в массиве
    masked_array = arr[non_nan]
    f_interploate = interpolate.interp1d(non_nan, masked_array)

    # Производим процедуру интерполяции
    x = np.arange(0, len(arr))
    y = f_interploate(x)
    return(y)

# Проверка точности восстановления исходного ряда
### Input:
# parameter (str)        --- название столбца в датафрейме data, параметр, из которого сотавляется временной ряд
# mask (str)             --- название столбца в датафрейме data, который содержит бинарный код маски пропусков
# data (pd DataFrame)    --- датафрейм, в котором содержится вся необходимая информация
# withoutgap_arr (array) --- массив без пропусков
### Output:
# Функция выводит на экран значения трех метрик: MAE, RMSE, MedianAE
def validate(parameter, mask, data, withoutgap_arr, gap_value = -100.0):

    # Исходный массив
    arr_parameter = np.array(data[parameter])
    # Масссив с пропуском
    arr_mask = np.array(data[mask])
    # В каких элементах присутствуют пропуски
    ids_gaps = np.ravel(np.argwhere(arr_mask == -100.0))
    ids_non_gaps = np.ravel(np.argwhere(arr_mask != -100.0))

    true_values = arr_parameter[ids_gaps]
    predicted_values = withoutgap_arr[ids_gaps]

    print('Совокупный размер пропусков:', len(true_values))
    print(f'Общая длина временного ряда: {len(arr_parameter)}')
    min_value = min(true_values)
    max_value = max(true_values)
    print('Минимальное значение в пропуске - ', min_value)
    print('Максимальное значение в пропуске- ', max_value)

    # Выводим на экран метрики
    MAE = mean_absolute_error(true_values, predicted_values)
    print('Mean absolute error -', round(MAE, 4))

    RMSE = (mean_squared_error(true_values, predicted_values)) ** 0.5
    print('RMSE -', round(RMSE, 4))

    MedianAE = median_absolute_error(true_values, predicted_values)
    print('Median absolute error -', round(MedianAE, 4))

    mape = mean_absolute_percentage_error(true_values, predicted_values)
    print('MAPE -', round(mape, 4), '\n')

    # Массив с пропусками
    array_gaps = np.ma.masked_where(arr_mask == gap_value, arr_mask)

    plt.plot(data['Date'], arr_parameter, c='green', alpha=0.5, label='Actual values')
    plt.plot(data['Date'], withoutgap_arr, c='red', alpha=0.5, label='Predicted values')
    plt.plot(data['Date'], array_gaps, c='blue', alpha=1.0)
    plt.ylabel('Sea level, m', fontsize=15)
    plt.xlabel('Date', fontsize=15)
    plt.grid()
    plt.legend(fontsize=15)
    plt.show()



folder_to_save = 'D:/iccs_article/linear'

# Заполнение пропусков и проверка результатов
for file in ['Synthetic.csv', 'Sea_hour.csv', 'Sea_10_240.csv']:
    print(file)
    data = pd.read_csv(file)
    data['Date'] = pd.to_datetime(data['Date'])
    dataframe = data.copy()

    start = timeit.default_timer()

    withoutgap_arr = linear_interpolation(np.array(data['gap']))
    print('Runtime -', timeit.default_timer() - start)
    dataframe['gap'] = withoutgap_arr
    validate(parameter = 'Height', mask = 'gap', data = data, withoutgap_arr = withoutgap_arr)

    save_path = os.path.join(folder_to_save, file)
    # Create folder if it doesnt exists
    if os.path.isdir(folder_to_save) == False:
        os.makedirs(folder_to_save)
    dataframe.to_csv(save_path)

