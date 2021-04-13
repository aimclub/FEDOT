import os
import pandas as pd
import numpy as np

from gapfilling.validation_and_metrics import *

from pylab import rcParams
rcParams['figure.figsize'] = 18, 7

from fedot.utilities.ts_gapfilling import SimpleGapFiller


def run_spline(folder_with_filled, files_list,
               columns_with_gap, file_with_results, vis=False):
    """
    The function starts the report creation for Kalman filter gap-filling results

    :param folder_to_save: where to save csv files with filled gaps
    :param files_list: list with file name, which will be processed
    :param columns_with_gap: list with names of columns with gaps
    :param file_with_results: file with final report with metrics
    :param vis: is there a need to make visualisations
    """

    mapes = []
    for file_id, file in enumerate(files_list):
        # Source dataframe with gaps in it
        data = pd.read_csv(os.path.join('..', 'data', file))
        data['Date'] = pd.to_datetime(data['Date'])

        # Already filled dataframe
        data_filled = pd.read_csv(os.path.join(folder_with_filled, file))
        data_filled['Date'] = pd.to_datetime(data_filled['Date'])

        # Creating the dataframe
        mini_dataframe = pd.DataFrame({'File': [file] * 6,
                                       'Metric': ['MAE', 'RMSE', 'MedAE',
                                                  'MAPE', 'Min gap value',
                                                  'Max gap value']})
        # For every gap series
        for column_with_gap in columns_with_gap:
            print(f'File - {file}, column with gaps - {column_with_gap}')

            array_with_gaps = np.array(data[column_with_gap])
            withoutgap_arr = np.array(data_filled[column_with_gap])

            min_val, max_val, mae, rmse, medianae, mape = validate(parameter='Height',
                                                                   mask=column_with_gap,
                                                                   data=data,
                                                                   withoutgap_arr=withoutgap_arr,
                                                                   vis=vis)

            mini_dataframe[column_with_gap] = [mae, rmse, medianae, mape, min_val, max_val]
            mapes.append(mape)

        print(mini_dataframe)
        print('\n')

        if file_id == 0:
            main_dataframe = mini_dataframe
        else:
            frames = [main_dataframe, mini_dataframe]
            main_dataframe = pd.concat(frames)

    mapes = np.array(mapes)
    print(f'Mean MAPE value - {np.mean(mapes):.4f}')

    path_to_save = os.path.dirname(os.path.abspath(file_with_results))
    if os.path.isdir(path_to_save) == False:
        os.makedirs(path_to_save)
    main_dataframe.to_csv(file_with_results, index=False)


# Run the kalman report creation
folder_with_filled = '../data/spline'
files_list = ['Synthetic.csv', 'Sea_hour.csv', 'Sea_10_240.csv', 'Temperature.csv', 'Traffic.csv']
columns_with_gap = ['gap', 'gap_center']
file_with_results = '../data/reports/spline_report.csv'

if __name__ == '__main__':
    run_spline(folder_with_filled, files_list,
               columns_with_gap, file_with_results, vis=False)
