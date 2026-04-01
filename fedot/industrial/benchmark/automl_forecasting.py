# Map from: https://github.com/rakshitha123/TSForecasting/
import csv
import math
import os
import platform
import time
from datetime import datetime
from distutils.util import strtobool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gmean, pearsonr, spearmanr
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, median_absolute_error,
                             mean_squared_error, r2_score)

FREQUENCY_MAP = {
    '4_seconds': '4S',
    'minutely': '1min',
    '10_minutes': '10min',
    '5_minutes': '5min',
    'half_hourly': '30min',
    'hourly': '1H',
    'daily': '1D',
    'weekly': '1W',
    'monthly': '1M',
    'quarterly': '1Q',
    'yearly': '1Y'
}

# Found using code in findFrequencies.R
# Credit to Mitchell O'Hara-Wild and Rob J Hyndman for the R code
# Link: https://github.com/robjhyndman/forecast

frequencies = {
    'economics_1.csv': 7,
    'economics_10.csv': 1,
    'economics_100.csv': 6,
    'economics_11.csv': 2,
    'economics_12.csv': 4,
    'economics_13.csv': 1,
    'economics_14.csv': 1,
    'economics_15.csv': 1,
    'economics_16.csv': 1,
    'economics_17.csv': 12,
    'economics_18.csv': 12,
    'economics_19.csv': 6,
    'economics_2.csv': 6,
    'economics_20.csv': 6,
    'economics_21.csv': 499,
    'economics_22.csv': 12,
    'economics_23.csv': 12,
    'economics_24.csv': 4,
    'economics_25.csv': 4,
    'economics_26.csv': 2,
    'economics_27.csv': 1,
    'economics_28.csv': 12,
    'economics_29.csv': 6,
    'economics_3.csv': 1,
    'economics_30.csv': 12,
    'economics_31.csv': 12,
    'economics_32.csv': 53,
    'economics_33.csv': 12,
    'economics_34.csv': 333,
    'economics_35.csv': 4,
    'economics_36.csv': 12,
    'economics_37.csv': 12,
    'economics_38.csv': 1,
    'economics_39.csv': 125,
    'economics_4.csv': 1,
    'economics_40.csv': 12,
    'economics_41.csv': 12,
    'economics_42.csv': 12,
    'economics_43.csv': 12,
    'economics_44.csv': 12,
    'economics_45.csv': 6,
    'economics_46.csv': 4,
    'economics_47.csv': 1,
    'economics_48.csv': 4,
    'economics_49.csv': 1,
    'economics_5.csv': 8,
    'economics_50.csv': 4,
    'economics_51.csv': 2,
    'economics_52.csv': 1,
    'economics_53.csv': 1,
    'economics_54.csv': 1,
    'economics_55.csv': 1,
    'economics_56.csv': 6,
    'economics_57.csv': 1,
    'economics_58.csv': 6,
    'economics_59.csv': 4,
    'economics_6.csv': 1,
    'economics_60.csv': 6,
    'economics_61.csv': 4,
    'economics_62.csv': 1,
    'economics_63.csv': 12,
    'economics_64.csv': 1,
    'economics_65.csv': 12,
    'economics_66.csv': 1,
    'economics_67.csv': 12,
    'economics_68.csv': 1,
    'economics_69.csv': 12,
    'economics_7.csv': 6,
    'economics_70.csv': 4,
    'economics_71.csv': 4,
    'economics_72.csv': 4,
    'economics_73.csv': 2,
    'economics_74.csv': 4,
    'economics_75.csv': 4,
    'economics_76.csv': 4,
    'economics_77.csv': 4,
    'economics_78.csv': 4,
    'economics_79.csv': 4,
    'economics_8.csv': 1,
    'economics_80.csv': 4,
    'economics_81.csv': 4,
    'economics_82.csv': 2,
    'economics_83.csv': 2,
    'economics_84.csv': 2,
    'economics_85.csv': 2,
    'economics_86.csv': 2,
    'economics_87.csv': 4,
    'economics_88.csv': 4,
    'economics_89.csv': 4,
    'economics_9.csv': 8,
    'economics_90.csv': 12,
    'economics_91.csv': 12,
    'economics_92.csv': 6,
    'economics_93.csv': 6,
    'economics_94.csv': 1,
    'economics_95.csv': 77,
    'economics_96.csv': 6,
    'economics_97.csv': 4,
    'economics_98.csv': 4,
    'economics_99.csv': 4,
    'finance_1.csv': 1,
    'finance_10.csv': 22,
    'finance_100.csv': 1,
    'finance_11.csv': 8,
    'finance_12.csv': 1,
    'finance_13.csv': 1,
    'finance_14.csv': 12,
    'finance_15.csv': 12,
    'finance_16.csv': 1,
    'finance_17.csv': 11,
    'finance_18.csv': 13,
    'finance_19.csv': 1,
    'finance_2.csv': 1,
    'finance_20.csv': 1,
    'finance_21.csv': 14,
    'finance_22.csv': 9,
    'finance_23.csv': 1,
    'finance_24.csv': 1,
    'finance_25.csv': 1,
    'finance_26.csv': 1,
    'finance_27.csv': 4,
    'finance_28.csv': 7,
    'finance_29.csv': 6,
    'finance_3.csv': 1,
    'finance_30.csv': 12,
    'finance_31.csv': 333,
    'finance_32.csv': 200,
    'finance_33.csv': 6,
    'finance_34.csv': 1,
    'finance_35.csv': 1,
    'finance_36.csv': 6,
    'finance_37.csv': 2,
    'finance_38.csv': 4,
    'finance_39.csv': 2,
    'finance_4.csv': 3,
    'finance_40.csv': 4,
    'finance_41.csv': 4,
    'finance_42.csv': 1,
    'finance_43.csv': 4,
    'finance_44.csv': 13,
    'finance_45.csv': 14,
    'finance_46.csv': 16,
    'finance_47.csv': 17,
    'finance_48.csv': 3,
    'finance_49.csv': 1,
    'finance_5.csv': 1,
    'finance_50.csv': 7,
    'finance_51.csv': 4,
    'finance_52.csv': 1,
    'finance_53.csv': 1,
    'finance_54.csv': 3,
    'finance_55.csv': 2,
    'finance_56.csv': 1,
    'finance_57.csv': 1,
    'finance_58.csv': 16,
    'finance_59.csv': 1,
    'finance_6.csv': 12,
    'finance_60.csv': 166,
    'finance_61.csv': 1,
    'finance_62.csv': 10,
    'finance_63.csv': 1,
    'finance_64.csv': 1,
    'finance_65.csv': 12,
    'finance_66.csv': 16,
    'finance_67.csv': 1,
    'finance_68.csv': 10,
    'finance_69.csv': 1,
    'finance_7.csv': 7,
    'finance_70.csv': 10,
    'finance_71.csv': 12,
    'finance_72.csv': 15,
    'finance_73.csv': 1,
    'finance_74.csv': 1,
    'finance_75.csv': 3,
    'finance_76.csv': 7,
    'finance_77.csv': 7,
    'finance_78.csv': 1,
    'finance_79.csv': 7,
    'finance_8.csv': 7,
    'finance_80.csv': 1,
    'finance_81.csv': 3,
    'finance_82.csv': 3,
    'finance_83.csv': 7,
    'finance_84.csv': 1,
    'finance_85.csv': 1,
    'finance_86.csv': 3,
    'finance_87.csv': 6,
    'finance_88.csv': 1,
    'finance_89.csv': 1,
    'finance_9.csv': 10,
    'finance_90.csv': 8,
    'finance_91.csv': 4,
    'finance_92.csv': 1,
    'finance_93.csv': 1,
    'finance_94.csv': 3,
    'finance_95.csv': 3,
    'finance_96.csv': 1,
    'finance_97.csv': 3,
    'finance_98.csv': 4,
    'finance_99.csv': 6,
    'human_1.csv': 12,
    'human_10.csv': 143,
    'human_100.csv': 10,
    'human_11.csv': 8,
    'human_12.csv': 1,
    'human_13.csv': 7,
    'human_14.csv': 200,
    'human_15.csv': 12,
    'human_16.csv': 1,
    'human_17.csv': 12,
    'human_18.csv': 3,
    'human_19.csv': 7,
    'human_2.csv': 24,
    'human_20.csv': 100,
    'human_21.csv': 24,
    'human_22.csv': 24,
    'human_23.csv': 24,
    'human_24.csv': 24,
    'human_25.csv': 24,
    'human_26.csv': 24,
    'human_27.csv': 24,
    'human_28.csv': 24,
    'human_29.csv': 24,
    'human_3.csv': 91,
    'human_30.csv': 125,
    'human_31.csv': 7,
    'human_32.csv': 12,
    'human_33.csv': 24,
    'human_34.csv': 6,
    'human_35.csv': 3,
    'human_36.csv': 12,
    'human_37.csv': 7,
    'human_38.csv': 1,
    'human_39.csv': 12,
    'human_4.csv': 1,
    'human_40.csv': 3,
    'human_41.csv': 24,
    'human_42.csv': 24,
    'human_43.csv': 24,
    'human_44.csv': 24,
    'human_45.csv': 24,
    'human_46.csv': 24,
    'human_47.csv': 24,
    'human_48.csv': 24,
    'human_49.csv': 24,
    'human_5.csv': 125,
    'human_50.csv': 24,
    'human_51.csv': 24,
    'human_52.csv': 24,
    'human_53.csv': 6,
    'human_54.csv': 100,
    'human_55.csv': 24,
    'human_56.csv': 6,
    'human_57.csv': 12,
    'human_58.csv': 12,
    'human_59.csv': 12,
    'human_6.csv': 333,
    'human_60.csv': 6,
    'human_61.csv': 12,
    'human_62.csv': 12,
    'human_63.csv': 6,
    'human_64.csv': 1,
    'human_65.csv': 24,
    'human_66.csv': 24,
    'human_67.csv': 24,
    'human_68.csv': 1,
    'human_69.csv': 3,
    'human_7.csv': 125,
    'human_70.csv': 2,
    'human_71.csv': 2,
    'human_72.csv': 2,
    'human_73.csv': 2,
    'human_74.csv': 3,
    'human_75.csv': 2,
    'human_76.csv': 6,
    'human_77.csv': 5,
    'human_78.csv': 8,
    'human_79.csv': 8,
    'human_8.csv': 143,
    'human_80.csv': 9,
    'human_81.csv': 7,
    'human_82.csv': 24,
    'human_83.csv': 24,
    'human_84.csv': 24,
    'human_85.csv': 24,
    'human_86.csv': 24,
    'human_87.csv': 24,
    'human_88.csv': 24,
    'human_89.csv': 24,
    'human_9.csv': 12,
    'human_90.csv': 24,
    'human_91.csv': 24,
    'human_92.csv': 24,
    'human_93.csv': 24,
    'human_94.csv': 24,
    'human_95.csv': 24,
    'human_96.csv': 24,
    'human_97.csv': 24,
    'human_98.csv': 12,
    'human_99.csv': 6,
    'nature_1.csv': 15,
    'nature_10.csv': 12,
    'nature_100.csv': 10,
    'nature_11.csv': 12,
    'nature_12.csv': 30,
    'nature_13.csv': 12,
    'nature_14.csv': 12,
    'nature_15.csv': 32,
    'nature_16.csv': 21,
    'nature_17.csv': 1,
    'nature_18.csv': 1,
    'nature_19.csv': 12,
    'nature_2.csv': 12,
    'nature_20.csv': 6,
    'nature_21.csv': 6,
    'nature_22.csv': 6,
    'nature_23.csv': 3,
    'nature_24.csv': 12,
    'nature_25.csv': 10,
    'nature_26.csv': 12,
    'nature_27.csv': 1,
    'nature_28.csv': 50,
    'nature_29.csv': 12,
    'nature_3.csv': 7,
    'nature_30.csv': 12,
    'nature_31.csv': 14,
    'nature_32.csv': 12,
    'nature_33.csv': 12,
    'nature_34.csv': 12,
    'nature_35.csv': 6,
    'nature_36.csv': 12,
    'nature_37.csv': 1,
    'nature_38.csv': 12,
    'nature_39.csv': 1,
    'nature_4.csv': 12,
    'nature_40.csv': 3,
    'nature_41.csv': 12,
    'nature_42.csv': 12,
    'nature_43.csv': 12,
    'nature_44.csv': 12,
    'nature_45.csv': 12,
    'nature_46.csv': 12,
    'nature_47.csv': 12,
    'nature_48.csv': 12,
    'nature_49.csv': 12,
    'nature_5.csv': 4,
    'nature_50.csv': 12,
    'nature_51.csv': 12,
    'nature_52.csv': 12,
    'nature_53.csv': 2,
    'nature_54.csv': 1,
    'nature_55.csv': 3,
    'nature_56.csv': 12,
    'nature_57.csv': 125,
    'nature_58.csv': 12,
    'nature_59.csv': 12,
    'nature_6.csv': 4,
    'nature_60.csv': 1,
    'nature_61.csv': 1,
    'nature_62.csv': 1,
    'nature_63.csv': 12,
    'nature_64.csv': 24,
    'nature_65.csv': 7,
    'nature_66.csv': 12,
    'nature_67.csv': 12,
    'nature_68.csv': 1,
    'nature_69.csv': 11,
    'nature_7.csv': 12,
    'nature_70.csv': 12,
    'nature_71.csv': 12,
    'nature_72.csv': 12,
    'nature_73.csv': 1,
    'nature_74.csv': 11,
    'nature_75.csv': 30,
    'nature_76.csv': 28,
    'nature_77.csv': 499,
    'nature_78.csv': 2,
    'nature_79.csv': 28,
    'nature_8.csv': 1,
    'nature_80.csv': 24,
    'nature_81.csv': 25,
    'nature_82.csv': 12,
    'nature_83.csv': 12,
    'nature_84.csv': 12,
    'nature_85.csv': 11,
    'nature_86.csv': 1,
    'nature_87.csv': 48,
    'nature_88.csv': 1,
    'nature_89.csv': 1,
    'nature_9.csv': 1,
    'nature_90.csv': 1,
    'nature_91.csv': 1,
    'nature_92.csv': 4,
    'nature_93.csv': 12,
    'nature_94.csv': 4,
    'nature_95.csv': 2,
    'nature_96.csv': 4,
    'nature_97.csv': 12,
    'nature_98.csv': 12,
    'nature_99.csv': 1,
}
REVERSE_FREQUENCY_MAP = {v: k for k, v in FREQUENCY_MAP.items()}


# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of
# the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(full_file_path_and_name, replace_missing_vals_with='NaN',
                             value_column_name='series_value'):
    try:
        with open(full_file_path_and_name, 'r', encoding='utf-8') as file:
            return parse_file(file, replace_missing_vals_with, value_column_name)
    except BaseException:
        with open(full_file_path_and_name, 'r', encoding='cp1252') as file:
            return parse_file(file, replace_missing_vals_with, value_column_name)


def parse_file(file, replace_missing_vals_with, value_column_name):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    for line in file:
        # Strip white space from start/end of line
        line = line.strip()

        if line:
            if line.startswith('@'):  # Read meta-data
                if not line.startswith('@data'):
                    line_content = line.split(' ')
                    if line.startswith('@attribute'):
                        if (
                                len(line_content) != 3
                        ):  # Attributes have both name and type
                            raise Exception('Invalid meta-data specification.')

                        col_names.append(line_content[1])
                        col_types.append(line_content[2])
                    else:
                        if (
                                len(line_content) != 2
                        ):  # Other meta-data have only values
                            raise Exception('Invalid meta-data specification.')

                        if line.startswith('@frequency'):
                            frequency = line_content[1]
                        elif line.startswith('@horizon'):
                            forecast_horizon = int(line_content[1])
                        elif line.startswith('@missing'):
                            contain_missing_values = bool(
                                strtobool(line_content[1])
                            )
                        elif line.startswith('@equallength'):
                            contain_equal_length = bool(strtobool(line_content[1]))

                else:
                    if len(col_names) == 0:
                        raise Exception(
                            'Missing attribute section. Attribute section must come before data.'
                        )

                    found_data_tag = True
            elif not line.startswith('#'):
                if len(col_names) == 0:
                    raise Exception(
                        'Missing attribute section. Attribute section must come before data.'
                    )
                elif not found_data_tag:
                    raise Exception('Missing @data tag.')
                else:
                    if not started_reading_data_section:
                        started_reading_data_section = True
                        found_data_section = True
                        all_series = []

                        for col in col_names:
                            all_data[col] = []

                    full_info = line.split(':')

                    if len(full_info) != (len(col_names) + 1):
                        raise Exception('Missing attributes/values in series.')

                    series = full_info[len(full_info) - 1]
                    series = series.split(',')

                    if len(series) == 0:
                        raise Exception(
                            'A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol'
                        )

                    numeric_series = []

                    for val in series:
                        if val == '?':
                            numeric_series.append(replace_missing_vals_with)
                        else:
                            numeric_series.append(float(val))

                    if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                    ):
                        raise Exception(
                            'All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series.'
                        )

                    all_series.append(pd.Series(numeric_series).array)

                    for i in range(len(col_names)):
                        att_val = None
                        if col_types[i] == 'numeric':
                            att_val = int(full_info[i])
                        elif col_types[i] == 'string':
                            att_val = str(full_info[i])
                        elif col_types[i] == 'date':
                            att_val = datetime.strptime(
                                full_info[i], '%Y-%m-%d %H-%M-%S'
                            )
                        else:
                            # Currently, the code supports only numeric, string and date types. Extend this as required.
                            raise Exception('Invalid attribute type.')

                        if att_val is None:
                            raise Exception('Invalid attribute value.')
                        else:
                            all_data[col_names[i]].append(att_val)

            line_count = line_count + 1

    if line_count == 0:
        raise Exception('Empty file.')
    if len(col_names) == 0:
        raise Exception('Missing attribute section.')
    if not found_data_section:
        raise Exception('Missing series information under data section.')

    all_data[value_column_name] = all_series
    loaded_data = pd.DataFrame(all_data)

    return (loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length)


class Utils:
    """Utility functions"""

    ignored_presets = [
        # 'preset-superfast__60_proc-1_limit-60.0', 'preset-superfast__1200_proc-1_limit-3.0',
        # 'preset-superfast__900_proc-1_limit-4.0', 'preset-superfast__600_proc-1_limit-6.0',
        # 'preset-superfast__600_proc-10_limit-6.0', 'preset-superfast__900_proc-10_limit-4.0',
        # 'preset-superfast__300_proc-1_limit-12.0', 'preset-superfast__300_proc-10_limit-12.0',
        # 'preset-fast__900_proc-1_limit-4.0', 'preset-fast__1200_proc-1_limit-3.0', 'preset-fast__600_proc-1_limit-6.0',
        # 'preset-fast__900_proc-10_limit-4.0', 'preset-fast__300_proc-1_limit-12.0',
        # 'preset-fast__600_proc-10_limit-6.0', 'preset-fast__60_proc-1_limit-60.0',
        # 'preset-fast__300_proc-10_limit-12.0', 'preset-fast_parallel__1200_proc-1_limit-3.0',
        # 'preset-default__900_proc-10_limit-4.0', 'preset-default__900_proc-1_limit-4.0',
        # 'preset-fast_parallel__600_proc-10_limit-6.0', 'preset-fast_parallel__900_proc-10_limit-4.0',
        # 'preset-default__1200_proc-1_limit-3.0', 'preset-default__600_proc-10_limit-6.0',
        # 'preset-fast_parallel__900_proc-1_limit-4.0', 'preset-fast_parallel__300_proc-10_limit-12.0',
        # 'preset-fast_parallel__600_proc-1_limit-6.0', 'preset-default__600_proc-1_limit-6.0',
        # 'preset-default__300_proc-10_limit-12.0', 'preset-fast_parallel__300_proc-1_limit-12.0',
        # 'preset-default__300_proc-1_limit-12.0', 'preset-fast_parallel__60_proc-1_limit-60.0',
        # 'preset-default__60_proc-1_limit-60.0', 'preset-all__600_proc-1_limit-6.0',
        # 'preset-all__900_proc-1_limit-4.0','preset-all__1200_proc-1_limit-3.0',
        # 'preset-all__300_proc-1_limit-12.0','preset-all__300_proc-10_limit-12.0',
        # 'preset-all_300_proc-1_limit-12.0', 'preset-all_600_proc-1_limit-6.0', 'preset-all_60_proc-1_limit-60.0',
        # 'preset-all__600_proc-10_limit-6.0', 'preset-all__900_proc-10_limit-4.0',
        # 'preset-default_300_proc-1_limit-12.0', 'preset-default_600_proc-1_limit-6.0',
        # 'preset-default_60_proc-1_limit-60.0', 'preset-all__60_proc-1_limit-60.0',
    ]

    @staticmethod
    def regression_scores(actual, predicted, y_train,
                          scores_dir=None,
                          forecaster_name=None,
                          multioutput='uniform_average',
                          **kwargs):
        """Calculate forecasting metrics and optionally save results.

        :param np.array actual: Original time series values
        :param np.array predicted: Predicted time series values
        :param np.array y_train: Training values (required for MASE)
        :param str scores_dir: Path to file to record scores (str or None), defaults to None
        :param str forecaster_name: Name of model (str)
        :param str multioutput: 'raw_values' (raw errors), 'uniform_average' (averaged errors), defaults to 'uniform_average'
        :raises TypeError: If forecaster_name is not provided when saving results to file
        :return results: Dictionary of results
        """

        # Convert pd.Series to NumPy Array
        if predicted.shape == (actual.shape[0], 1) and not pd.core.frame.DataFrame:
            predicted = predicted.flatten()

        if predicted.shape != actual.shape:
            raise ValueError(f'Predicted ({predicted.shape}) and actual ({actual.shape}) shapes do not match!')

        # mase = MeanAbsoluteScaledError(multioutput='uniform_average')
        pearson = Utils.correlation(actual, predicted, method='pearson')
        spearman = Utils.correlation(actual, predicted, method='spearman')

        results = {
            'MAE': mean_absolute_error(actual, predicted, multioutput=multioutput),
            'MAE2': median_absolute_error(actual, predicted),
            'MAEover': Utils.mae_over(actual, predicted),
            'MAEunder': Utils.mae_under(actual, predicted),
            'MAPE': mean_absolute_percentage_error(actual, predicted, multioutput=multioutput),
            # 'MASE': mase(actual, predicted, y_train=y_train),
            'ME': np.mean(actual - predicted),
            'MSE': mean_squared_error(actual, predicted, multioutput=multioutput),
            'Pearson Correlation': pearson[0],
            'Pearson P-value': pearson[1],
            'R2': r2_score(actual, predicted, multioutput=multioutput),
            'RMSE': math.sqrt(mean_squared_error(actual, predicted, multioutput=multioutput)),
            'sMAPE': Utils.smape(actual, predicted),
            'Spearman Correlation': spearman[0],
            'Spearman P-value': spearman[1],
        }

        # Grimes calls for "maximizing the geometric mean of (−MAE) and average daily Spearman correlation"
        # This must be an error, as you cannot calculate geometric mean with negative numbers. This uses
        # geometric mean of MAE and (1-SRC) with the intention of minimizing the metric.
        results['GM-MAE-SR'] = Utils.geometric_mean(results['MAE'], results['Spearman Correlation'])
        results['GM-MASE-SR'] = Utils.geometric_mean(results['MASE'], results['Spearman Correlation'])

        if 'duration' in kwargs.keys():
            results['duration'] = kwargs['duration']

        if scores_dir is not None:
            if forecaster_name is None:
                raise TypeError('Forecaster name required to save scores')
            os.makedirs(scores_dir, exist_ok=True)

            results = {
                **results,
                'environment': f'python_{platform.python_version()}-os_{platform.system()}',
                'device': f'node_{platform.node()}-pro_{platform.processor()}',
            }

            Utils.write_to_csv(os.path.join(scores_dir, f'{forecaster_name}.csv'), results)

        return results

    @staticmethod
    def geometric_mean(error_score, rank_correlation_score):
        """Calculates the geometric mean of some mean error score and a mean rank correlation score.

        :param float error_score: Mean error score
        :param float rank_score: Mean rank correlation score
        :return float: Geometric mean of error and 1-rank scores
        """
        # Grimes calls for "maximizing the geometric mean of (−MAE) and average daily Spearman correlation"
        # It is not possible to calculate geometric mean with negative numbers without some conversion.
        # Therefore, this work uses geometric mean of MAE and (1-SRC) with the intention of minimizing the metric.
        return gmean([error_score, 1 - rank_correlation_score])

    @staticmethod
    def geometric_mean_MAE_SR(actual, predicted):
        """Calculates the geometric mean of MAE and a Spearman correlation score.

        :param np.array actual: Real values
        :param np.array predicted: Predicted values
        :return float: Geometric mean of MAE and SRC
        """
        MAE = mean_absolute_error(actual, predicted, multioutput='uniform_average'),
        SRC = Utils.correlation(actual, predicted, method='spearman')[0]
        return Utils.geometric_mean([MAE, SRC])

    @staticmethod
    def correlation(actual, predicted, method='pearson'):
        """Wrapper to extract correlations and p-values from scipy

        :param np.array actual: Actual values
        :param np.array predicted: Predicted values
        :param str method: Correlation type, defaults to 'pearson'
        :raises ValueError: If unknown correlation method is passed
        :return: Correlation (float) and pvalue (float)
        """
        if method == 'pearson':
            result = pearsonr(actual, predicted)
        elif method == 'spearman':
            result = spearmanr(actual, predicted)
        else:
            raise ValueError(f'Unknown correlation method: {method}')

        try:
            correlation = result.correlation
            pvalue = result.pvalue
        except BaseException:  # older scipy versions returned a tuple instead of an object
            correlation = result[0]
            pvalue = result[1]

        return correlation, pvalue

    @staticmethod
    def mae_over(actual, predicted):
        """Overestimated predictions (from Grimes et al. 2014)"""
        errors = predicted - actual
        positive_errors = np.clip(errors, 0, errors.max())
        return np.mean(positive_errors)

    @staticmethod
    def mae_under(actual, predicted):
        """Underestimated predictions (from Grimes et al. 2014)"""
        errors = predicted - actual
        negative_errors = np.clip(errors, errors.min(), 0)
        return np.absolute(np.mean(negative_errors))

    @staticmethod
    def smape(actual, predicted):
        """sMAPE"""
        return 100 / len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))

    @staticmethod
    def write_to_csv(path, results):
        """Record modelling results in a CSV file.

        :param str path: the result file path
        :param dict results: a dict containing results from running a model
        """

        np.set_printoptions(precision=4)

        # Remove unneeded values
        # unused_cols = []
        # for col in unused_cols:
        #     if col in results.keys():
        #         del results[col]

        if len(results) > 0:
            HEADERS = sorted(list(results.keys()), key=lambda v: str(v).upper())
            if 'model' in HEADERS:
                HEADERS.insert(0, HEADERS.pop(HEADERS.index('model')))

            for key, value in results.items():
                if value is None or value == '':
                    results[key] = 'None'

            try:
                Utils._write_to_csv(path, results, HEADERS)
            except OSError:
                # try a second time: permission error can be due to Python not
                # having closed the file fast enough after the previous write
                time.sleep(1)  # in seconds
                Utils._write_to_csv(path, results, HEADERS)

    @staticmethod
    def _write_to_csv(path, results, headers):
        """Open and write results to CSV file.

        :param str path: Path to file
        :param dict results: Values to write
        :param list headers: A list of strings to order values by
        """

        is_new_file = not os.path.exists(path)
        with open(path, 'a+', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            if is_new_file:
                writer.writerow(headers)
            writer.writerow([results[header] for header in headers])

    @staticmethod
    def plot_forecast(actual, predicted, results_subdir, forecaster_name):
        """Plot forecasted vs actual values

        :param np.array actual: Original time series values
        :param np.array predicted: Forecasted values
        :param str results_subdir: Path to output directory
        :param str forecaster_name: Model name
        """
        plt.clf()
        pd.plotting.register_matplotlib_converters()

        # Create plot
        plt.figure(0, figsize=(20, 3))  # Pass plot ID to prevent memory issues
        plt.plot(actual, label='actual')
        plt.plot(predicted, label='predicted')
        save_path = os.path.join(results_subdir, 'plots', f'{forecaster_name}.png')
        os.makedirs(os.path.join(results_subdir, 'plots'), exist_ok=True)
        Utils.save_plot(forecaster_name, save_path=save_path)

    @staticmethod
    def save_plot(title,
                  xlabel=None,
                  ylabel=None,
                  suptitle='',
                  show=False,
                  legend=None,
                  save_path=None,
                  yscale='linear'):
        """Apply title and axis labels to plot. Show and save to file. Clear plot.

        :param title: Title for plot
        :param xlabel: Plot X-axis label
        :param ylabel: Plot Y-axis label
        :param title: Subtitle for plot
        :param show: Show plot on screen, defaults to False
        :param legend: Legend, defaults to None
        :param save_path: Save plot to file if not None, defaults to None
        :param yscale: Y-Scale ('linear' or 'log'), defaults to 'linear'
        """

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        plt.yscale(yscale)

        plt.title(title)
        plt.suptitle(suptitle)

        if legend is not None:
            plt.legend(legend, loc='upper left')

        # Show plot
        if show:
            plt.show()
        # Show plot as file
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')

        # Clear for next plot
        plt.cla()
        plt.clf()
        plt.close('all')

    @staticmethod
    def get_csv_datasets(datasets_directory):
        """Fetch list of file names of CSV datasets

        :param str datasets_directory: Path to datasets directory
        :raises NotADirectoryError: If datasets_directory does not exist
        :raises IOError: If datasets_directory does not have CSV files
        :return: list of dataset file names
        """

        if not os.path.exists(datasets_directory):
            raise NotADirectoryError('Datasets direcotry path does not exist')

        csv_files = [
            f for f in os.listdir(datasets_directory)
            if f.endswith('csv') and '0_metadata' not in f
        ]

        if len(csv_files) == 0:
            raise IOError('No CSV files found')

        return csv_files

    @staticmethod
    def split_test_set(test_df, horizon):
        """Split test dataset into list of smaller sets for rolling origin forecasting

        :param pd.DataFrame test_df: Test dataset
        :param int horizon: Forecasting horizon
        :return: List of DataFrame objects
        """

        test_splits = []
        total = 0  # total length of test splits
        for _ in range(0, len(test_df) - 1, horizon):  # The -1 is because the last split may be less than horizon
            try:
                test_splits.append(test_df.iloc[total:total + horizon, :])
            except BaseException:  # If 1D (series)
                test_splits.append(test_df.iloc[total:total + horizon])
            total += horizon

        # Leftover rows
        if total < len(test_df):
            test_splits.append(test_df.tail(len(test_df) - total))
        return test_splits

    @staticmethod
    def summarize_dataset_results(results_dir, plots=True):
        """Analyse results saved to file

        :param str results_subdir: Path to relevant results directory
        :param bool plots: Save plots as images, defaults to True
        """

        stats_dir = os.path.join(results_dir, 'statistics')

        test_results = []
        failed = []
        dataset = os.path.basename(os.path.normpath(results_dir))

        # For each library/preset, get mean scores
        for library in os.listdir(results_dir):
            subdir = os.path.join(results_dir, library)
            for preset in os.listdir(subdir):
                preset_dir = os.path.join(subdir, preset)
                if subdir == stats_dir:
                    continue

                scores_path = os.path.join(preset_dir, f'{library}.csv')
                failed_path = os.path.join(preset_dir, f'failed.txt')

                if os.path.exists(scores_path):
                    df = pd.read_csv(scores_path, index_col=False)
                    test_results.append({'library': library, 'preset': preset, 'file': dataset, 'failed': 0,
                                         'num_iterations': len(df),
                                         **df.mean(numeric_only=True).to_dict()})
                elif os.path.exists(failed_path):
                    failed.append({'library': library, 'preset': preset, 'file': dataset, 'failed': 1})
                else:
                    raise FileNotFoundError(f'Results file(s) missing in {preset_dir}')

        os.makedirs(stats_dir, exist_ok=True)

        # Combine scores into one CSV file
        test_scores = pd.DataFrame(test_results)
        if len(test_scores) > 0:
            failed = pd.DataFrame(failed)
            test_scores = pd.concat([test_scores, failed])

        # Save all scores as CSV
        output_file = os.path.join(stats_dir, '1_all_scores.csv')
        test_scores.to_csv(output_file, index=False)

        # Scores per library across all presets and failed training counts
        if len(test_scores) > 0:
            summarized_scores = Utils.save_latex(test_scores, output_file.replace('csv', 'tex'))
            if plots:
                Utils.save_heatmap(summarized_scores, os.path.join(stats_dir, 'heatmap.csv'),
                                   os.path.join(stats_dir, 'heatmap.png'))
                # Utils.plot_test_scores(test_scores, stats_dir, plots)

    @staticmethod
    def save_latex(df, output_file):
        """Save dataframe of results in a LaTeX file

        :param pd.DataFrame df: Results
        :param str output_file: Path to .tex file
        """
        # Sort by GM-MAE-SR
        df = df.sort_values('GM-MAE-SR')
        # Filter columns
        df = df[['library', 'preset', 'duration', 'GM-MAE-SR', 'MAE', 'MASE', 'MSE', 'RMSE',
                 'Spearman Correlation']]
        # Rename columns
        df.columns = ['Library', 'Preset', 'Duration (sec.)', 'GM-MAE-SR', 'MAE', 'MASE', 'MSE', 'RMSE',
                      'SRC']
        df['Library'] = df['Library'].str.capitalize()  # Format library names
        df['Preset'] = df['Preset'] \
            .str.replace('Fedot', 'FEDOT') \
            .str.replace('Flaml', 'FLAML') \
            .str.replace('Autokeras', 'AutoKeras') \
            .str.replace('Autogluon', 'AutoGluon') \
            .str.replace('preset-', '') \
            .str.replace('proc-1', '') \
            .str.replace('proc-10', '') \
            .str.replace('-limit-3600', '') \
            .str.replace('-limit-3240', '') \
            .str.replace('-limit-3564', '') \
            .str.replace('-limit-57', '') \
            .str.replace('-limit-12', '') \
            .str.replace('-limit-60', '') \
            .str.replace('-limit-6', '') \
            .str.replace('_', ' ') \
            .str.capitalize() \
            .str.replace(' ', '-') \
            .str.replace('--', '-')

        # Save all scores as TEX
        df.style.format(precision=2, thousands=',', decimal='.').to_latex(
            output_file.replace('csv', 'tex'),
            caption='Test Scores Ordered by GM-MAE-SR',
            environment='table*',
            hrules=True,
            label='tab:summarized_scores',
            multirow_align='t',
            position='!htbp',
        )
        return df

    @staticmethod
    def plot_test_scores(test_scores, stats_dir, plots):
        """Plot test scores

        :param pd.DataFrame test_scores: Test scores
        :param str stats_dir: Path to output directory
        :param bool plots: If True, generate plots
        """
        test_scores['library'] = test_scores['library'].str.capitalize()

        # Ignore deprecated/unused presets
        test_scores = test_scores[~test_scores['preset'].isin(Utils.ignored_presets)]

        # Save overall scores and generate plots
        if plots:
            # Bar plot of failed training attempts
            test_scores.plot.bar(y='failed', figsize=(35, 10))
            save_path = os.path.join(stats_dir, '3_failed_counts.png')
            Utils.save_plot('Failed Counts', save_path=save_path)

            # Boxplots
            for col, filename, title in [
                ('MAE', '5_MAE_box.png', 'MAE'),
                ('MSE', '5_MSE_box.png', 'MSE'),
                ('RMSE', '6_RMSE_box.png', 'RMSE'),
                ('Spearman Correlation', '6_Spearman_Correlation_box.png', 'Spearman Correlation'),
                ('duration', '8_duration_box.png', 'Duration (sec)'),
            ]:
                test_scores.boxplot(col, by='library')
                save_path = os.path.join(stats_dir, filename)
                Utils.save_plot(title, save_path=save_path)

        # Save mean scores and generate plots
        df_failed = test_scores[['library', 'failed']]
        df_failed = df_failed.set_index('library')
        df_failed = df_failed.groupby('library').sum()

        def plot_averages(group_col, cols_to_drop, fig_width, fig_height):
            df_grouped = test_scores.drop(cols_to_drop, axis=1).groupby(group_col)

            df_grouped.index = df_grouped[group_col]
            df_max = df_grouped.max()
            df_min = df_grouped.min()
            df_mean = df_grouped.mean()

            df_max.columns = [f'{c}_max' for c in df_max.columns.tolist()]
            df_min.columns = [f'{c}_min' for c in df_min.columns.tolist()]
            df_mean.columns = [f'{c}_mean' for c in df_mean.columns.tolist()]

            mean_scores = pd.concat([df_failed, df_max, df_min, df_mean], axis=1)

            output_file = os.path.join(stats_dir, f'3_mean_scores_by_{group_col}.csv')
            mean_scores.to_csv(output_file)

            if plots:
                # Bar plot of failed training attempts
                mean_scores.plot.bar(y='failed', figsize=(fig_width, fig_height))
                save_path = os.path.join(stats_dir, f'3_failed_counts_by_{group_col}.png')
                Utils.save_plot(f'Failed Counts by {group_col}', save_path=save_path, legend=None, yscale='linear')

                # Boxplots
                for col, filename, title, yscale in [
                    ('R2_mean', f'4_R2_mean_by_{group_col}.png', 'Mean R2', 'linear'),
                    ('MAE_mean', f'5_MAE_mean_by_{group_col}.png', 'Mean MAE', 'linear'),
                    ('MSE_mean', f'6_MSE_mean_by_{group_col}.png', 'Mean MSE', 'linear'),
                    ('duration_mean', f'7_duration_mean_by_{group_col}.png', 'Mean Duration', 'linear'),
                ]:
                    mean_scores.plot.bar(y=col, figsize=(fig_width, fig_height))
                    save_path = os.path.join(stats_dir, filename)
                    Utils.save_plot(title, save_path=save_path, legend=None, yscale=yscale)

        # Plot mean scores by library
        plot_averages(group_col='library',
                      cols_to_drop=['file', 'failed', 'preset'],
                      fig_width=6,
                      fig_height=3)

        # # Plot mean scores by library/preset
        # test_scores['library-preset'] = test_scores['library'] + ': ' + test_scores['preset']
        # plot_averages(group_col='library-preset',
        #               cols_to_drop=['file', 'failed', 'preset', 'library'],
        #               fig_width=35,
        #               fig_height=10)

    @staticmethod
    def summarize_overall_results(results_dir, forecast_type, plots=True):
        """Analyse results saved to file

        :param str results_subdir: Path to relevant results directory
        :param bool plots: Save plots as images, defaults to True
        """

        dataframes = []
        results_subdir = os.path.join(results_dir, f'{forecast_type}_forecasting')
        for dirpath, _, filenames in os.walk(results_subdir):
            if 'statistics' in dirpath and len(filenames) > 0:
                all_scores_path = os.path.join(dirpath, '1_all_scores.csv')
                try:
                    df = pd.read_csv(all_scores_path)
                    dataframes.append(df)
                except pd.errors.EmptyDataError as _:
                    print(_)

        all_scores = pd.concat(dataframes, axis=0)

        if len(all_scores) == 0:
            print('No results found. Skipping')
        else:
            stats_dir = os.path.join(results_dir, f'{forecast_type}_statistics')
            os.makedirs(stats_dir, exist_ok=True)

            # Save overall scores as CSV and TEX
            overall_scores_path = os.path.join(stats_dir, '1_all_scores.csv')
            all_scores.to_csv(overall_scores_path, index=False)
            summarized_scores = Utils.save_latex(all_scores, overall_scores_path.replace('csv', 'tex'))

            if plots:
                Utils.save_heatmap(summarized_scores, os.path.join(stats_dir, 'metrics_corr_heatmap.csv'),
                                   os.path.join(stats_dir, 'heatmap.png'))
                Utils.plot_test_scores(all_scores, stats_dir, plots)

    def save_heatmap(df, csv_path, png_path):
        """Save Pearson Correlation Matrix of metrics

        :param pd.DataFrame df: Results
        :param str csv_path: Path to CSV file
        :param str png_path: Path to PNG file
        """
        # Save Perason correlation heatmap of metrics as an indication of agreement.
        # columns = ['GM-MAE-SR', 'MAE', 'MASE', 'MSE', 'RMSE', 'SRC'] # SRC is not normally distributed
        columns = ['GM-MAE-SR', 'MAE', 'MASE', 'MSE', 'RMSE']  # SRC removed
        df[columns].to_csv('variables.csv')
        heatmap = df[columns].corr(method='pearson')

        # Save correlations and corresponding p-values as CSV
        heatmap.to_csv(csv_path)  # Save correlations as CSV
        heatmap.to_latex(csv_path.replace('.csv', '.tex'))  # Save correlations as .tex
        try:
            def calculate_pvalues(x, y): return pearsonr(x, y).pvalue
            df[columns].corr(method=calculate_pvalues).to_csv(csv_path.replace('.csv', '_pvalues.csv'))
        # older scipy versions return a tuple instead of an object
        except BaseException:
            def calculate_pvalues(x, y): return pearsonr(x, y)[1]
            df[columns].corr(method=calculate_pvalues).to_csv(csv_path.replace('.csv', '_pvalues.csv'))

        # Save correlation heatmap as image
        axes = sns.heatmap(heatmap,
                           annot=True,
                           cbar=False,
                           cmap='viridis',
                           fmt='.2f',
                           #    xticklabels=columns,
                           #    yticklabels=columns,
                           annot_kws={'size': 11}
                           )
        axes.set_xticklabels(axes.get_xticklabels(), fontsize=11, rotation=45, ha='right')
        axes.set_yticklabels(axes.get_yticklabels(), fontsize=11, rotation=45, va='top')
        # axes.set_xticklabels(columns, fontsize=11, rotation=45, ha='right')
        # axes.set_yticklabels(columns, fontsize=11, rotation=45, va='top')
        plt.tight_layout()
        Utils.save_plot('Pearson Correlation Heatmap', save_path=png_path)
