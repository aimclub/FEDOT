from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


if __name__ == '__main__':
    folder = 'poly'

    # Массивы с пропусками
    main_hour = pd.read_csv('data/tsla.csv')
    main_day = pd.read_csv('data/Sea_10_240.csv')

    # Заполненные разными методами временные ряды
    hour_fedot = pd.read_csv('data/fedot_ridge/tsla.csv')
    day_fedot = pd.read_csv('data/fedot_ridge_inverse/Sea_10_240.csv')

    # Полнимомиальная аппроксимация
    hour_poly = pd.read_csv(f'data/{folder}/tsla.csv')
    day_poly = pd.read_csv(f'data/{folder}/Sea_10_240.csv')

    # Части временного ряда, которые были пропусками
    hour_gap = np.array(main_hour['gap'])
    day_gap = np.array(main_day['gap'])
    real_hour = np.array(main_hour['Height'])
    real_day = np.array(main_day['Height'])

    # Индексы этих элементов
    hour_gap_ids = np.ravel(np.argwhere(hour_gap == -100.0))
    day_gap_ids = np.ravel(np.argwhere(day_gap == -100.0))

    fedot_hour_filled = np.array(hour_fedot['gap'])
    fedot_day_filled = np.array(day_fedot['gap'])

    poly_hour_filled = np.array(hour_poly['gap'])
    poly_day_filled = np.array(day_poly['gap'])

    # Рисуем график
    array_fedot_hour = np.ma.masked_where(hour_gap != -100.0, fedot_hour_filled)
    array_fedot_day = np.ma.masked_where(day_gap != -100.0, fedot_day_filled)

    array_poly_hour = np.ma.masked_where(hour_gap != -100.0, poly_hour_filled)
    array_poly_day = np.ma.masked_where(day_gap != -100.0, poly_day_filled)

    plt.plot(range(len(real_hour)), real_hour, label='Real data')
    plt.plot(range(len(real_hour)), array_fedot_hour, label='FEDOT filled')
    plt.plot(range(len(real_hour)), array_poly_hour, label='Another method')
    plt.grid()
    plt.legend()
    plt.title('Hour data')
    plt.show()

    plt.plot(range(len(real_day)), real_day, label='Real data')
    plt.plot(range(len(real_day)), array_fedot_day, label='FEDOT filled')
    plt.plot(range(len(real_day)), array_poly_day, label='Another method')
    plt.grid()
    plt.legend()
    plt.title('Day data')
    plt.show()

    print('Значение метрик для часовых данных, метрика MAE')
    real_hour_data = np.ravel(real_hour[hour_gap_ids])
    fedot_hour_data = np.ravel(fedot_hour_filled[hour_gap_ids])
    poly_hour_data = np.ravel(poly_hour_filled[hour_gap_ids])

    mae_metric = mean_absolute_error(real_hour_data, fedot_hour_data)
    print(f'FEDOT MAE: {mae_metric:.2f}')
    mape_metric = mean_absolute_percentage_error(real_hour_data, fedot_hour_data)*100
    print(f'FEDOT MAPE: {mape_metric:.2f}')
    print(f'FEDOT SMAPE: {smape(real_hour_data, fedot_hour_data):.2f}')

    mae_metric = mean_absolute_error(real_hour_data, poly_hour_data)
    print(f'Another method MAE: {mae_metric:.2f}')
    mape_metric = mean_absolute_percentage_error(real_hour_data, poly_hour_data)*100
    print(f'Another method MAPE: {mape_metric:.2f}')
    print(f'Another method SMAPE: {smape(real_hour_data, poly_hour_data):.2f}')

    print('\nЗначение метрик для суточных данных, метрика MAE')
    real_day_data = np.ravel(real_day[day_gap_ids])
    fedot_day_data = np.ravel(fedot_day_filled[day_gap_ids])
    poly_day_data = np.ravel(poly_day_filled[day_gap_ids])

    mae_metric = mean_absolute_error(real_day_data, fedot_day_data)
    print(f'FEDOT MAE: {mae_metric:.2f}')
    mape_metric = mean_absolute_percentage_error(real_day_data, fedot_day_data)*100
    print(f'FEDOT MAPE: {mape_metric:.2f}')
    print(f'FEDOT SMAPE: {smape(real_day_data, fedot_day_data):.2f}')

    mae_metric = mean_absolute_error(real_day_data, poly_day_data)
    print(f'Another method MAE: {mae_metric:.2f}')
    mape_metric = mean_absolute_percentage_error(real_day_data, poly_day_data)*100
    print(f'Another method MAPE: {mape_metric:.2f}')
    print(f'FEDOT SMAPE: {smape(real_day_data, poly_day_data):.2f}')
