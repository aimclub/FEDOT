from typing import Union
import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum


class Fedot_data_helper():

    def autodetect_data_type(self,
                             task: Task) -> DataTypesEnum:
        if task.task_type == TaskTypesEnum.ts_forecasting:
            return DataTypesEnum.ts
        else:
            return DataTypesEnum.table

    def array_to_input_data(self,
                            features_array: np.array,
                            target_array: np.array,
                            task: Task = Task(TaskTypesEnum.classification)):
        data_type = self.autodetect_data_type(task)
        idx = np.arange(len(features_array))

        return InputData(idx=idx, features=features_array, target=target_array, task=task, data_type=data_type)

    def define_data(self,
                    ml_task: Task,
                    features: Union[str, np.ndarray, pd.DataFrame, InputData],
                    target: Union[str, np.ndarray, pd.Series] = None,
                    is_predict=False):
        if type(features) == InputData:
            # native FEDOT format for input data
            data = features
            data.task = ml_task
        elif type(features) == pd.DataFrame:
            # pandas format for input data
            if target is None:
                target = np.array([])

            if isinstance(target, str) and target in features.columns:
                target_array = features[target]
                del features[target]
            else:
                target_array = target

            data = self.array_to_input_data(features_array=np.asarray(features),
                                            target_array=np.asarray(target_array),
                                            task=ml_task)
        elif type(features) == np.ndarray:
            # numpy format for input data
            if target is None:
                target = np.array([])

            if isinstance(target, str):
                target_array = features[target]
                del features[target]
            else:
                target_array = target

            data = self.array_to_input_data(features_array=features,
                                            target_array=target_array,
                                            task=ml_task)
        elif type(features) == tuple:
            data = self.array_to_input_data(features_array=features[0],
                                            target_array=features[1],
                                            task=ml_task)
        elif type(features) == str:
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
                                          target_column=target,
                                          data_type=data_type)
        else:
            raise ValueError('Please specify a features as path to csv file or as Numpy array')

        return data
