from typing import Union

import numpy as np
import pandas as pd

from fedot.api.api_utils.data_definition import data_strategy_selector
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum


class ApiDataSources:
    """
    Class for selecting optimal data processing strategies based on type of data.
    Available data sources are:
        * numpy array
        * pandas DataFrame
        * string (path to csv file)
        * InputData (FEDOT dataclass)
    """

    def define_data(self,
                    ml_task: Task,
                    features: Union[str, np.ndarray, pd.DataFrame, InputData, dict],
                    target: Union[str, np.ndarray, pd.Series] = None,
                    is_predict=False):
        """ Prepare data for composing """
        try:
            # TODO remove workaround
            idx = None
            if isinstance(features, dict) and 'idx' in features:
                idx = features['idx']
                del features['idx']
            data = data_strategy_selector(features=features,
                                          target=target,
                                          ml_task=ml_task,
                                          is_predict=is_predict)
            if isinstance(data, dict) and idx is not None:
                for k in data.keys():
                    data[k].idx = idx
        except Exception as ex:
            raise ValueError(f'Please specify a features as path to csv file or as Numpy array: {ex}')
        return data

    def define_predictions(self,
                           task_type: TaskTypesEnum,
                           current_pipeline: Pipeline,
                           test_data: InputData):

        if task_type == TaskTypesEnum.classification:
            prediction = current_pipeline.predict(test_data, output_mode='labels')
            output_prediction = prediction
        elif task_type == TaskTypesEnum.ts_forecasting:
            # Convert forecast into one-dimensional array
            prediction = current_pipeline.predict(test_data)
            forecast = np.ravel(np.array(prediction.predict))
            prediction.predict = forecast
            output_prediction = prediction
        else:
            prediction = current_pipeline.predict(test_data)
            output_prediction = prediction

        return output_prediction
