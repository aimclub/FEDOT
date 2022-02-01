from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Union

import numpy as np
import pandas as pd

from fedot.core.data.data import InputData, array_to_input_data
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


class DataDefiner:

    def __init__(self, strategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy) -> None:
        self._strategy = strategy

    def define_data(self, features: Union[tuple, str, np.ndarray, pd.DataFrame, InputData],
                    ml_task: Task,
                    target: str = None,
                    is_predict: bool = False) -> None:
        return self._strategy.define_data(features,
                                          ml_task,
                                          target,
                                          is_predict)


class StrategyDefineData(ABC):
    @abstractmethod
    def define_data(self, features: Union[tuple, str, np.ndarray, pd.DataFrame, InputData],
                    ml_task: Task,
                    target: str = None,
                    is_predict: bool = False):
        pass


class FedotStrategy(StrategyDefineData):
    def define_data(self, features: InputData,
                    ml_task: Task,
                    target: str = None,
                    is_predict: bool = False) -> InputData:
        # InputData format for input data
        data = deepcopy(features)
        data.task = ml_task
        return data


class TupleStrategy(StrategyDefineData):
    def define_data(self, features: tuple,
                    ml_task: Task,
                    target: str = None,
                    is_predict: bool = False) -> InputData:
        data = array_to_input_data(features_array=features[0],
                                   target_array=features[1],
                                   task=ml_task)
        return data


class PandasStrategy(StrategyDefineData):
    """ Class for wrapping pandas DataFrames into InputData """

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


class NumpyStrategy(StrategyDefineData):
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


class CsvStrategy(StrategyDefineData):
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


class MulitmodalStrategy(StrategyDefineData):
    def define_data(self, features: dict,
                    ml_task: str,
                    target: str = None,
                    is_predict: bool = False,
                    idx=None) -> MultiModalData:
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
    strategy_dict = {InputData: FedotStrategy(),
                     MultiModalData: FedotStrategy(),
                     tuple: TupleStrategy(),
                     pd.DataFrame: PandasStrategy(),
                     np.ndarray: NumpyStrategy(),
                     str: CsvStrategy(),
                     dict: MulitmodalStrategy()}

    data = DataDefiner(strategy_dict[data_type])
    return data.define_data(features, ml_task, target, is_predict)
