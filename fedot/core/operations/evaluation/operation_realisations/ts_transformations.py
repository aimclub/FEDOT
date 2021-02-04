from copy import copy
from typing import List

import pandas as pd
import numpy as np

from fedot.core.repository.tasks import TaskTypesEnum


class LaggedTransformation:
    """ Realisation of lagged transformation for time series forecasting"""

    def __init__(self, **params):
        self.params = params

    def fit(self, data):
        """ Class doesn't support fit operation

        :param : Input data for training transformation operation
        """
        #self._appropriate_task_check(data.task)

    def transform(self, data, is_fit_chain_stage: bool):
        target = data.target
        parameters = data.task.task_params

        if is_fit_chain_stage:
            print("Преобразование выполняется для fit'a")
            # Convert data to lagged form
            lagged_dataframe = pd.DataFrame({'target': target})
            vals = lagged_dataframe['target']
            for i in range(1, parameters.max_window_size+1):
                frames = [lagged_dataframe, vals.shift(i)]
                lagged_dataframe = pd.concat(frames, axis=1)

            # Remove incomplete rows
            lagged_dataframe.dropna(inplace=True)

            transformed = np.array(lagged_dataframe)

            # Generate dataset with features
            features_columns = transformed[:, 1:]
            features_columns = np.fliplr(features_columns)
        else:
            print("Преобразование выполняется для predict'a")
            features = np.array(data.features)
            features_columns = features[-parameters.max_window_size:]
            features_columns = features_columns.reshape(1, -1)

        return features_columns

    @staticmethod
    def _appropriate_task_check(task):
        if task.task_type == TaskTypesEnum.ts_forecasting:
            pass
        else:
            raise ValueError(f'LaggedTransformation operation available only for time series task')


def lagged_data_mapping(input_data, is_fit_chain_stage: bool):
    """
    Функция для обновления данных

    :param input_data: Input data for mapping operation
    """
    print(f'Стадия {is_fit_chain_stage}')

    task_params = input_data.task.task_params

    # For time series forecasting we need to transform target size due to
    window_size = task_params.max_window_size

    # Multi-target transformation
    if task_params.forecast_length > 1:
        # Target transformation
        print(f'Исходная (изначальная) длина {len(input_data.target)}')
        current_target = input_data.target[window_size:]
        print(f'Обрезанный вектор с предсказаниями {len(current_target)}')
        df = pd.DataFrame({'target': current_target})
        vals = df['target']
        for i in range(1, task_params.forecast_length):
            frames = [df, vals.shift(-i)]
            df = pd.concat(frames, axis=1)

        # Remove incomplete rows
        df.dropna(inplace=True)
        targets = np.array(df)
        print(f'Преобразованный target {targets.shape}')
        features = np.array(input_data.features[
                            :(-task_params.forecast_length + 1)])
        print(f'Преобразованный features {features.shape}')
        input_data.target = targets
        input_data.features = features
        input_data.idx = range(0, len(targets))
    else:
        input_data.idx = range(0, len(input_data.features))
        input_data.target = input_data.target[window_size:]

    return input_data
