from abc import ABC, abstractmethod
from copy import deepcopy
from os import PathLike
from typing import Union, Optional

import numpy as np
import pandas as pd

from fedot.core.data.data import InputData, array_to_input_data, features_datetime_to_int
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

FeaturesType = Union[str, PathLike, np.ndarray, pd.DataFrame, InputData, MultiModalData, dict, tuple]
TargetType = Union[str, PathLike, np.ndarray, pd.Series, dict]


class DataDefiner:

    def __init__(self, strategy: 'StrategyDefineData') -> None:
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy) -> None:
        self._strategy = strategy

    def define_data(self, features: FeaturesType,
                    task: Task,
                    target: Optional[str] = None,
                    is_predict: bool = False) -> Union[InputData, MultiModalData]:
        return self._strategy.define_data(features,
                                          task,
                                          target,
                                          is_predict)


class StrategyDefineData(ABC):
    @abstractmethod
    def define_data(self, features: Union[tuple, str, np.ndarray, pd.DataFrame, InputData],
                    task: Task,
                    target: str = None,
                    is_predict: bool = False) -> Union[InputData, MultiModalData]:
        pass


class FedotStrategy(StrategyDefineData):
    def define_data(self, features: Union[InputData, MultiModalData],
                    task: Task,
                    target: str = None,
                    is_predict: bool = False) -> Union[InputData, MultiModalData]:
        # InputData or MultiModalData format for input data
        data = deepcopy(features)
        data.task = task
        return data


class TupleStrategy(StrategyDefineData):
    def define_data(self, features: tuple,
                    task: Task,
                    target: str = None,
                    is_predict: bool = False) -> InputData:
        features[0][:] = features_datetime_to_int(features[0])

        data = array_to_input_data(features_array=features[0],
                                   target_array=features[1],
                                   task=task)
        return data


class PandasStrategy(StrategyDefineData):
    """ Class for wrapping pandas DataFrames into InputData """

    def define_data(self, features: pd.DataFrame,
                    task: Task,
                    target: str = None,
                    is_predict: bool = False) -> InputData:
        if target is None:
            target = np.array([])

        if isinstance(target, str) and target in features.columns:
            target_array = features[target]
            features = features.drop(columns=target)
        else:
            target_array = target

        features = features_datetime_to_int(features)

        data = array_to_input_data(features_array=np.asarray(features),
                                   target_array=np.asarray(target_array),
                                   task=task)
        return data


class NumpyStrategy(StrategyDefineData):
    def define_data(self, features: np.ndarray,
                    task: Task,
                    target: Optional[int] = None,
                    is_predict: bool = False) -> InputData:
        # numpy format for input data
        if target is None:
            target = np.array([])

        if task.task_type is TaskTypesEnum.ts_forecasting:
            target_array = features
        elif isinstance(target, int):
            target_array = features[target]
            features = np.delete(features, target, axis=1)
        else:
            target_array = target

        features = features_datetime_to_int(features)

        data = array_to_input_data(features_array=features,
                                   target_array=target_array,
                                   task=task)
        return data


class CsvStrategy(StrategyDefineData):
    def define_data(self, features: Union[str, PathLike],
                    task: Task,
                    target: str = None,
                    is_predict: bool = False) -> InputData:
        # CSV files as input data, by default - table data

        if task.task_type == TaskTypesEnum.ts_forecasting:
            # For time series forecasting format - time series
            data = InputData.from_csv_time_series(task=task,
                                                  file_path=features,
                                                  target_column=target,
                                                  is_predict=is_predict)
        else:
            # Make default features table
            # CSV files as input data
            data = InputData.from_csv(features, task=task,
                                      target_columns=target,
                                      data_type=DataTypesEnum.table)
        return data


class MultimodalStrategy(StrategyDefineData):
    """
    Gets dict of NumPy arrays or InputData sources as input data
    and returns MultiModalData object with source names defined by data type
    """
    source_name_by_type = {'table': 'data_source_table',
                           'ts': 'data_source_ts',
                           'multi_ts': 'data_source_ts',
                           'text': 'data_source_text',
                           'image': 'data_source_image'}

    def define_data(self, features: dict,
                    task: Task,
                    target: str = None,
                    is_predict: bool = False,
                    idx=None) -> MultiModalData:

        # change data type to InputData
        for inner_data in features.values():
            if not isinstance(inner_data, InputData):
                converted_data = features_datetime_to_int(inner_data)
                inner_data[:] = array_to_input_data(features_array=converted_data, target_array=target,
                                                    task=task, idx=idx)
        # create labels for data sources
        sources = dict((f'{self.source_name_by_type.get(features[data_part_key].data_type.name)}/{data_part_key}',
                        data_part)
                       for (data_part_key, data_part) in features.items())
        data = MultiModalData(sources)
        return data


def data_strategy_selector(features: FeaturesType, target: Optional[str] = None, task: Task = None,
                           is_predict: bool = None) -> Union[InputData, MultiModalData]:
    strategy = [strategy for cls, strategy in _strategy_dispatch.items() if isinstance(features, cls)][0]

    data = DataDefiner(strategy())
    return data.define_data(features, task, target, is_predict)


_strategy_dispatch = {InputData: FedotStrategy,
                      MultiModalData: FedotStrategy,
                      tuple: TupleStrategy,
                      pd.DataFrame: PandasStrategy,
                      np.ndarray: NumpyStrategy,
                      str: CsvStrategy,
                      PathLike: CsvStrategy,
                      dict: MultimodalStrategy}
