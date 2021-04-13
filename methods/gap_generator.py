import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 18, 7

import warnings
warnings.filterwarnings('ignore')


def generate_gaps(csv_file, gap_dict, gap_value, column_name, vis = False):
    """
    Function for generating gaps of a given duration in the selected places

    :param csv_file: path to the csv data file
    :param gap_dict: a dictionary with gaps, where the key is the index in the
    time series from which the omission will begin. The key value is the length
    of the gap. -1 in the value means that a skip is generated until the end of
    the row
    :param gap_value: flag for gaps
    :param column_name: name of the column with gaps
    :param vis: bool, is visualization required or not
    """

    dataframe = pd.read_csv(csv_file)
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    print(f'Total length of time series {len(dataframe)}')

    sea_level = np.array(dataframe['Height'])
    keys = list(gap_dict.keys())
    for key in keys:
        gap_size = gap_dict.get(key)
        if gap_size == -1:
            sea_level[key:] = gap_value
        else:
            sea_level[key:(key+gap_size)] = gap_value

    if vis:
        masked_array = np.ma.masked_where(sea_level == -100.0, sea_level)
        plt.plot(dataframe['Date'], dataframe['Height'], c='blue', alpha=0.3)
        plt.plot(dataframe['Date'], masked_array, c='blue')
        plt.xlabel('Date', fontsize=13)
        plt.grid()
        plt.show()

    # Save file
    dataframe[column_name] = sea_level
    gap_ids = np.ravel(np.argwhere(sea_level == gap_value))
    print(f'Total amount of gap elements {len(gap_ids)} '
          f'and ratio {(len(gap_ids)/len(dataframe))*100:.2f}\n')

    dataframe.to_csv(csv_file, index=False)


csv_file = 'data/Traffic.csv'
# 30%
generate_gaps(csv_file=csv_file,
              gap_dict={550: 150,
                        1000: 140,
                        1600: 360,
                        2500: 620,
                        4050: 420,
                        5400: 200},
              gap_value=-100.0,
              column_name='gap',
              vis=True)

generate_gaps(csv_file=csv_file,
              gap_dict={2500: 1500},
              gap_value=-100.0,
              column_name='gap_center',
              vis=True)
