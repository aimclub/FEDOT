from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from matplotlib import pyplot as plt
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate metric - Mean Absolute Percentage Error (MAPE)

    :param y_true: actual values
    :param y_pred: predicted values

    :return : MAPE value
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # The formula below has a drawback-if there is at least one value of 0.0 in the y_true array,
    # then by the formula np. mean(np. abs((y_true - y_pred) / y_true)) * 100 we get inf, so

    zero_indexes = np.argwhere(y_true == 0.0)
    for index in zero_indexes:
        # 0 values are replaced with very small ones
        y_true[index] = 0.01
    value = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return value


def validate(parameter, mask, data, withoutgap_arr, gap_value=-100.0, vis=False):
    """
    The function counts quality metrics and displays them on the screen

    :param parameter: name of the column in the dataframe, the parameter from which the time series is created
    :param mask: name of the column in the data dataframe that contains the gaps
    :param data: a dataframe to process
    :param withoutgap_arr: array without gaps
    :param gap_value: gap identifier in the array
    :param vis: is there a need to make visualisations
    """

    # Source array
    arr_parameter = np.array(data[parameter])
    # Array with gaps
    arr_mask = np.array(data[mask])
    # Ids of elements with gaps
    ids_gaps = np.ravel(np.argwhere(arr_mask == -100.0))
    ids_non_gaps = np.ravel(np.argwhere(arr_mask != -100.0))

    true_values = arr_parameter[ids_gaps]
    predicted_values = withoutgap_arr[ids_gaps]
    # print('Amount of gap elements:', len(true_values))
    # print(f'Total length of time series: {len(arr_parameter)}')
    min_value = min(true_values)
    max_value = max(true_values)
    # print('Minimum value in the gap - ', min_value)
    # print('Maximum value in the gap - ', max_value)

    # Display the metrics on the screen
    mae = mean_absolute_error(true_values, predicted_values)
    # print('Mean absolute error -', round(mae, 4))

    rmse = (mean_squared_error(true_values, predicted_values)) ** 0.5
    # print('RMSE -', round(rmse, 4))

    median_ae = median_absolute_error(true_values, predicted_values)
    # print('Median absolute error -', round(median_ae, 4))

    mape = mean_absolute_percentage_error(true_values, predicted_values)
    # print('MAPE -', round(mape, 4), '\n')

    # Array with gaps
    array_gaps = np.ma.masked_where(arr_mask == gap_value, arr_mask)

    if vis:
        plt.plot(data['Date'], arr_parameter, c='green', alpha=0.5, label='Actual values')
        plt.plot(data['Date'], withoutgap_arr, c='red', alpha=0.5, label='Predicted values')
        plt.plot(data['Date'], array_gaps, c='blue', alpha=1.0)
        plt.ylabel('Sea level, m', fontsize=15)
        plt.xlabel('Date', fontsize=15)
        plt.grid()
        plt.legend(fontsize=15)
        plt.show()

    return min_value, max_value, mae, rmse, median_ae, mape