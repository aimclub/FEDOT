import os
import pandas as pd
import numpy as np

from gapfilling.validation_and_metrics import *

from pylab import rcParams
rcParams['figure.figsize'] = 18, 7

folder_with_df = './data/reports'
df_list = ['batch_poly_report.csv', 'fedot_ridge_inverse_report.csv', 'fedot_ridge_report.csv',
           'kalman_report.csv', 'linear_report.csv', 'ma_report.csv', 'poly_report.csv', 'spline_report.csv']

for df_name in df_list:
    print(f'\nPROCESSING {df_name} ...')
    dataframe = pd.read_csv(os.path.join(folder_with_df, df_name))

    syn = []
    meteo = []
    economic = []
    for ts_name in dataframe['File'].unique():
        ts_dataframe = dataframe[dataframe['File'] == ts_name]
        df_with_mape = ts_dataframe[ts_dataframe['Metric'] == 'MAPE']

        mapes = []
        for gap_type in ['gap', 'gap_center']:
            mape_val = float(df_with_mape[gap_type])
            mapes.append(mape_val)

        # Calculate mean value
        mapes = np.array(mapes)
        mean_mape = float(np.mean(mapes))

        if ts_name == 'Synthetic.csv':
            syn.append(round(mean_mape, 1))
        elif ts_name == 'Sea_hour.csv':
            # Waiting
            sea_hour_value = round(mean_mape, 2)
        elif ts_name == 'Sea_10_240.csv':
            sea_daily_value = round(mean_mape, 2)
        elif ts_name == 'Temperature.csv':
            tmp_value = round(mean_mape, 2)
            mean_value = (sea_daily_value + sea_hour_value + tmp_value) / 3
            mean_value = round(mean_value, 1)
            meteo.append(mean_value)
        elif ts_name == 'Traffic.csv':
            economic.append(round(mean_mape, 1))

    print(f'Synthetic data - {syn}')
    print(f'Meteodata - {meteo}')
    print(f'Economic - {economic}')

