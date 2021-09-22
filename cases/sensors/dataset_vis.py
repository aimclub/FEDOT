import warnings
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')


def show_tep_dataset():
    """ Display as plots all time series, which are in TEP
    (Tennessee Eastman Process) Dataset

    Link to the dataset: https://paperswithcode.com/dataset/tep
    """
    tep_path = '../data/time_series/tep_data.csv'
    tep_df = pd.read_csv(tep_path, header=None)

    for column in tep_df.columns:
        plt.plot(tep_df[column])
        plt.grid()
        plt.show()


if __name__ == '__main__':
    show_tep_dataset()
