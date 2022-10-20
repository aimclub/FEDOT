from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union, Optional

import numpy as np
import pandas as pd

from fedot.core.data.data import InputData, array_to_input_data
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

FeaturesType = Union[str, np.ndarray, pd.DataFrame, InputData, dict, tuple]
TargetType = Union[str, np.ndarray, pd.Series, dict]


class DataDefiner:

    def __init__(self, strategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy) -> None:
        self._strategy = strategy

    def define_data(self, features: FeaturesType,
                    ml_task: Task,
                    target: Optional[str] = None,
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
    def define_data(self, features: Union[InputData, MultiModalData],
                    ml_task: Task,
                    target: str = None,
                    is_predict: bool = False) -> Union[InputData, MultiModalData]:
        # InputData or MultiModalData format for input data
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
            features = features.drop(columns=[target])
        else:
            target_array = target

        data = array_to_input_data(features_array=np.asarray(features),
                                   target_array=np.asarray(target_array),
                                   task=ml_task)
        return data


class NumpyStrategy(StrategyDefineData):
    def define_data(self, features: np.ndarray,
                    ml_task: Task,
                    target: Optional[int] = None,
                    is_predict: bool = False) -> InputData:
        # numpy format for input data
        if target is None:
            target = np.array([])

        if isinstance(target, int):
            target_array = features[target]
            features = np.delete(features, target, axis=1)
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
            data = InputData.from_csv(features, task=ml_task,
                                      target_columns=target,
                                      data_type=data_type)
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
                    ml_task: Task,
                    target: str = None,
                    is_predict: bool = False,
                    idx=None) -> MultiModalData:

        # change data type to InputData
        for source in features:
            if not isinstance(features[source], InputData):
                features[source] = array_to_input_data(features_array=features[source], target_array=target,
                                                       task=ml_task, idx=idx)
        # create labels for data sources
        sources = dict((f'{self.source_name_by_type.get(features[data_part_key].data_type.name)}/{data_part_key}',
                        data_part)
                       for (data_part_key, data_part) in features.items())
        data = MultiModalData(sources)
        return data


def data_strategy_selector(features, target, ml_task: Task = None, is_predict: bool = None):
    data_type = type(features)

    data = DataDefiner(_strategy_dispatch[data_type]())
    return data.define_data(features, ml_task, target, is_predict)


_strategy_dispatch = {InputData: FedotStrategy,
                      MultiModalData: FedotStrategy,
                      tuple: TupleStrategy,
                      pd.DataFrame: PandasStrategy,
                      np.ndarray: NumpyStrategy,
                      str: CsvStrategy,
                      dict: MultimodalStrategy}
