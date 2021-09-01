from __future__ import annotations
from abc import ABC, abstractmethod
from functools import partial
from typing import Union
import pandas as pd
import numpy as np

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task


def autodetect_data_type(task: Task) -> DataTypesEnum:
    if task.task_type == TaskTypesEnum.ts_forecasting:
        return DataTypesEnum.ts
    else:
        return DataTypesEnum.table


def array_to_input_data(features_array: np.array,
                        target_array: np.array,
                        task: Task = Task(TaskTypesEnum.classification)):
    data_type = autodetect_data_type(task)
    idx = np.arange(len(features_array))

    return InputData(idx=idx, features=features_array, target=target_array, task=task, data_type=data_type)


class Data_definer:

    def __init__(self, strategy: Strategy_to_define_data) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> Strategy_to_define_data:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy_to_define_data) -> None:
        self._strategy = strategy

    def define_data(self, features: Union[tuple, str, np.ndarray, pd.DataFrame, InputData],
                    ml_task: Task,
                    target: str = None,
                    is_predict: bool = False) -> None:
        return self._strategy.define_data(features,
                                          ml_task,
                                          target,
                                          is_predict)


class Strategy_to_define_data(ABC):
    @abstractmethod
    def define_data(self, features: Union[tuple, str, np.ndarray, pd.DataFrame, InputData],
                    ml_task: Task,
                    target: str = None,
                    is_predict: bool = False):
        pass


class Fedot_strategy(Strategy_to_define_data):
    def define_data(self, features: InputData,
                    ml_task: Task,
                    target: str = None,
                    is_predict: bool = False) -> InputData:
        # pandas format for input data
        data = features
        data.task = ml_task
        return data


class Tuple_strategy(Strategy_to_define_data):
    def define_data(self, features: tuple,
                    ml_task: Task,
                    target: str = None,
                    is_predict: bool = False) -> InputData:
        data = array_to_input_data(features_array=features[0],
                                   target_array=features[1],
                                   task=ml_task)
        return data


class Pandas_strategy(Strategy_to_define_data):
    def define_data(self, features: pd.DataFrame,
                    ml_task: Task,
                    target: str = None,
                    is_predict: bool = False) -> InputData:
        if target is None:
            target = np.array([])

        if isinstance(target, str) and target in features.columns:
            target_array = features[target]
            del features[target]
        else:
            target_array = target

        data = array_to_input_data(features_array=np.asarray(features),
                                   target_array=np.asarray(target_array),
                                   task=ml_task)
        return data


class Numpy_strategy(Strategy_to_define_data):
    def define_data(self, features: np.ndarray,
                    ml_task: Task,
                    target: str = None,
                    is_predict: bool = False) -> InputData:
        # numpy format for input data
        if target is None:
            target = np.array([])

        if isinstance(target, str):
            target_array = features[target]
            del features[target]
        else:
            target_array = target

        data = array_to_input_data(features_array=features,
                                   target_array=target_array,
                                   task=ml_task)
        return data


class CSV_strategy(Strategy_to_define_data):
    def define_data(self, features: str,
                    ml_task: Task,
                    target: str = None,
                    is_predict: bool = False) -> InputData:
        # CSV files as input data, by default - table data
        if target is None:
            target = 'target'

        data_type = DataTypesEnum.table
        if ml_task.task_type == TaskTypesEnum.ts_forecasting:
            # For time series forecasting format - time series
            data = InputData.from_csv_time_series(task=ml_task,
                                                  file_path=features,
                                                  target_column=target,
                                                  is_predict=is_predict)
        else:
            # Make default features table
            # CSV files as input data
            if target is None:
                target = 'target'
            data = InputData.from_csv(features, task=ml_task,
                                      target_columns=target,
                                      data_type=data_type)
        return data


class Mulitmodal_strategy(Strategy_to_define_data):
    def define_data(self, features: dict,
                    ml_task: str,
                    target: str = None,
                    is_predict: bool = False) -> MultiModalData:
        if target is None:
            target = np.array([])
        target_array = target

        data_part_transformation_func = partial(array_to_input_data, target_array=target_array, task=ml_task)

        # create labels for data sources
        sources = dict((f'data_source_ts/{data_part_key}', data_part_transformation_func(features_array=data_part))
                       for (data_part_key, data_part) in features.items())
        data = MultiModalData(sources)
        return data


def data_strategy_selector(features, target, ml_task: Task = None, is_predict: bool = None):
    data_type = type(features)
    strategy_dict = {InputData: Fedot_strategy(),
                     tuple: Tuple_strategy(),
                     pd.DataFrame: Pandas_strategy(),
                     np.ndarray: Numpy_strategy(),
                     str: CSV_strategy(),
                     dict: Mulitmodal_strategy()}

    data = Data_definer(strategy_dict[data_type])
    return data.define_data(features, ml_task, target, is_predict)
