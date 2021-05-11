import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedot.core.data.load_data import TextBatchLoader
from fedot.core.data.merge import DataMerger
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@dataclass
class Data:
    """
    Base Data type class
    """
    idx: np.array
    features: np.array
    task: Task
    data_type: DataTypesEnum

    @staticmethod
    def from_csv(file_path=None,
                 delimiter=',',
                 task: Task = Task(TaskTypesEnum.classification),
                 data_type: DataTypesEnum = DataTypesEnum.table,
                 columns_to_drop: Optional[List] = None,
                 target_column: Optional[str] = ''):
        """
        :param file_path: the path to the CSV with data
        :param columns_to_drop: the names of columns that should be dropped
        :param delimiter: the delimiter to separate the columns
        :param task: the task that should be solved with data
        :param data_type: the type of data interpretation
        :param target_column: name of target column (last column if empty and no target if None)
        :return:
        """

        data_frame = pd.read_csv(file_path, sep=delimiter)
        if columns_to_drop:
            data_frame = data_frame.drop(columns_to_drop, axis=1)
        data_frame = _convert_dtypes(data_frame=data_frame)
        data_array = np.array(data_frame).T
        idx = data_array[0]

        if target_column == '':
            target_column = data_frame.columns[-1]

        if target_column and target_column in data_frame.columns:
            target = np.array(data_frame[target_column]).astype(np.float)
            pos = list(data_frame.keys()).index(target_column)
            features = np.delete(data_array.T, [0, pos], axis=1)
        else:
            # no target in data
            features = data_array[1:].T
            target = None

        return InputData(idx=idx, features=features, target=target, task=task, data_type=data_type)

    @staticmethod
    def from_csv_time_series(task: Task,
                             file_path=None,
                             delimiter=',',
                             is_predict=False,
                             target_column: Optional[str] = ''):
        data_frame = pd.read_csv(file_path, sep=delimiter)
        time_series = np.array(data_frame[target_column])
        if is_predict:
            # Prepare data for prediction
            len_forecast = task.task_params.forecast_length

            start_forecast = len(time_series)
            end_forecast = start_forecast + len_forecast
            input_data = InputData(idx=np.arange(start_forecast, end_forecast),
                                   features=time_series,
                                   target=None,
                                   task=task,
                                   data_type=DataTypesEnum.ts)
        else:
            # Prepare InputData for train the chain
            input_data = InputData(idx=np.arange(0, len(time_series)),
                                   features=time_series,
                                   target=time_series,
                                   task=task,
                                   data_type=DataTypesEnum.ts)

        return input_data

    @staticmethod
    def from_image(images: Union[str, np.ndarray] = None,
                   labels: Union[str, np.ndarray] = None,
                   task: Task = Task(TaskTypesEnum.classification)):
        """
        :param images: the path to the directory with image data in np.ndarray format or array in np.ndarray format
        :param labels: the path to the directory with image labels in np.ndarray format or array in np.ndarray format
        :param task: the task that should be solved with data
        :return:
        """
        features = images
        target = labels

        if type(images) is str:
            features = np.load(images)
            target = np.load(labels)

        idx = np.arange(0, len(features))

        return InputData(idx=idx, features=features, target=target, task=task, data_type=DataTypesEnum.image)

    @staticmethod
    def from_text_meta_file(meta_file_path: str = None,
                            label: str = 'label',
                            task: Task = Task(TaskTypesEnum.classification),
                            data_type: DataTypesEnum = DataTypesEnum.text):

        if os.path.isdir(meta_file_path):
            raise ValueError("""CSV file expected but got directory""")

        df_text = pd.read_csv(meta_file_path)
        df_text = df_text.sample(frac=1).reset_index(drop=True)
        messages = df_text['text'].astype('U').tolist()

        features = np.array(messages)
        target = np.array(df_text[label])
        idx = [index for index in range(len(target))]

        return InputData(idx=idx, features=features,
                         target=target, task=task, data_type=data_type)

    @staticmethod
    def from_text_files(files_path: str,
                        label: str = 'label',
                        task: Task = Task(TaskTypesEnum.classification),
                        data_type: DataTypesEnum = DataTypesEnum.text):

        if os.path.isfile(files_path):
            raise ValueError("""Path to the directory expected but got file""")

        df_text = TextBatchLoader(path=files_path).extract()

        features = np.array(df_text['text'])
        target = np.array(df_text[label])
        idx = [index for index in range(len(target))]

        return InputData(idx=idx, features=features,
                         target=target, task=task, data_type=data_type)


@dataclass
class InputData(Data):
    """
    Data class for input data for the nodes
    """
    target: Optional[np.array] = None

    @property
    def num_classes(self) -> Optional[int]:
        if self.task.task_type == TaskTypesEnum.classification:
            return len(np.unique(self.target))
        else:
            return None

    @staticmethod
    def from_predictions(outputs: List['OutputData']):
        if len(set([output.task.task_type for output in outputs])) > 1:
            raise ValueError('Inconsistent task types')

        task = outputs[0].task
        data_type = outputs[0].data_type

        # Update not only features but idx and target also
        idx, features, target = DataMerger(outputs).merge()

        return InputData(idx=idx, features=features, target=target, task=task,
                         data_type=data_type)

    def subset(self, start: int, end: int):
        if not (0 <= start <= end <= len(self.idx)):
            raise ValueError('Incorrect boundaries for subset')
        new_features = None
        if self.features is not None:
            new_features = self.features[start:end + 1]
        return InputData(idx=self.idx[start:end + 1], features=new_features,
                         target=self.target[start:end + 1], task=self.task, data_type=self.data_type)


@dataclass
class OutputData(Data):
    """
    Data type for data prediction in the node
    """
    predict: np.array = None
    target: Optional[np.array] = None


def _split_time_series(data, task):
    """ Split time series data into train and test parts

    :param data: array with data to split (not InputData)
    :param task: task to solve
    """

    input_features = data.features
    input_target = data.target
    forecast_length = task.task_params.forecast_length

    # Source time series divide into two parts
    x_train = input_features[:-forecast_length]
    x_test = input_features[:-forecast_length]

    y_train = x_train
    y_test = input_target[-forecast_length:]

    idx_for_train = np.arange(0, len(x_train))

    # Define indices for test
    start_forecast = len(x_train)
    end_forecast = start_forecast + forecast_length
    idx_for_predict = np.arange(start_forecast, end_forecast)

    # Prepare data to train the operation
    train_data = InputData(idx=idx_for_train,
                           features=x_train,
                           target=y_train,
                           task=task,
                           data_type=DataTypesEnum.ts)

    test_data = InputData(idx=idx_for_predict,
                          features=x_test,
                          target=y_test,
                          task=task,
                          data_type=DataTypesEnum.ts)

    return train_data, test_data


def _split_table(data, task, split_ratio, with_shuffle=False):
    """ Split table data into train and test parts

    :param data: array with data to split (not InputData)
    :param task: task to solve
    :param split_ratio: threshold for partitioning
    :param with_shuffle: is data needed to be shuffled or not
    """

    if not 0. < split_ratio < 1.:
        raise ValueError('Split ratio must belong to the interval (0; 1)')
    random_state = 42

    # Predictors and target
    input_features = data.features
    input_target = data.target

    x_train, x_test, y_train, y_test = train_test_split(input_features,
                                                        input_target,
                                                        test_size=1. - split_ratio,
                                                        shuffle=with_shuffle,
                                                        random_state=random_state)

    idx_for_train = np.arange(0, len(x_train))
    idx_for_predict = np.arange(0, len(x_test))

    # Prepare data to train the operation
    train_data = InputData(idx=idx_for_train,
                           features=x_train,
                           target=y_train,
                           task=task,
                           data_type=DataTypesEnum.table)

    test_data = InputData(idx=idx_for_predict,
                          features=x_test,
                          target=y_test,
                          task=task,
                          data_type=DataTypesEnum.table)

    return train_data, test_data


def train_test_data_setup(data: InputData, split_ratio=0.8,
                          shuffle_flag=False) -> Tuple[InputData, InputData]:
    """ Function for train and test split

    :param data: InputData for train and test splitting
    :param split_ratio: threshold for partitioning
    :param shuffle_flag: is data needed to be shuffled or not

    :return train_data: InputData for train
    :return test_data: InputData for validation
    """
    # Split into train and test
    if data.features is not None:
        task = data.task
        if data.data_type == DataTypesEnum.ts:
            train_data, test_data = _split_time_series(data, task)
        elif data.data_type == DataTypesEnum.table:
            train_data, test_data = _split_table(data, task, split_ratio,
                                                 with_shuffle=shuffle_flag)
        else:
            train_data, test_data = _split_table(data, task, split_ratio,
                                                 with_shuffle=shuffle_flag)
    else:
        raise ValueError('InputData must be not empty')

    return train_data, test_data


def _convert_dtypes(data_frame: pd.DataFrame):
    """ Function converts columns with objects into numerical form and fill na """
    objects: pd.DataFrame = data_frame.select_dtypes('object')
    for column_name in objects:
        warnings.warn(f'Automatic factorization for the column {column_name} with type "object" is applied.')
        encoded = pd.factorize(data_frame[column_name])[0]
        data_frame[column_name] = encoded
    data_frame = data_frame.fillna(0)
    return data_frame
