import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cases.sensors.launch_tools import make_forecasts


def run_experiment(horizons: list, validation_blocks: int, tuner_iterations: int):
    """ Launch experiments

    :param validation_blocks: number of parts for time series validation
    :param horizons: horizons for forecasting
    :param tuner_iterations: number of iterations for tuning
    """
    tep_path = '../data/time_series/tep_data.csv'
    tep_df = pd.read_csv(tep_path, header=None)

    for column in tep_df.columns:
        print(f'Process time series with id {column}')
        ts = np.array(tep_df[column])

        # Take last 3000 elements
        ts = ts[-3000:]
        # Launch validation for this time series
        make_forecasts(ts, column,
                       validation_blocks=validation_blocks,
                       horizons=horizons,
                       tuner_iterations=tuner_iterations,
                       save_path='../data/time_series/results')


if __name__ == '__main__':
    run_experiment(horizons=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                   tuner_iterations=2,
                   validation_blocks=3)
