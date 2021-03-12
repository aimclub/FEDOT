import os
import warnings
from copy import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedot.core.data.load_data import TextBatchLoader
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

        if target_column:
            target = np.array(data_frame[target_column]).astype(np.float)
            pos = list(data_frame.keys()).index(target_column)
            features = np.delete(data_array.T, [0, pos], axis=1)
        else:
            # no target in data
            features = data_array[1:].T
            target = None

        return InputData(idx=idx, features=features, target=target, task=task, data_type=data_type)

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
        target = df_text[label]
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

        features = df_text['text']
        target = df_text[label]
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

        dataset_merging_funcs = {
            DataTypesEnum.table: _combine_datasets_table,
            DataTypesEnum.ts: _combine_datasets_ts
        }
        dataset_merging_funcs.setdefault(data_type, _combine_datasets_table)

        # Update not only features but idx and target also
        idx, features, target = dataset_merging_funcs[data_type](outputs)

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
    Data type for data predicted in the node
    """
    predict: np.array = None
    target: Optional[np.array] = None


def split_train_test(data, split_ratio=0.8, with_shuffle=False, task: Task = None):
    assert 0. <= split_ratio <= 1.
    if task is not None and task.task_type == TaskTypesEnum.ts_forecasting:
        split_point = int(len(data) * split_ratio)
        # move pre-history of time series from train to test sample
        data_train, data_test = (data[:split_point],
                                 copy(data[split_point - task.task_params.max_window_size:]))
    else:
        if with_shuffle:
            data_train, data_test = train_test_split(data, test_size=1. - split_ratio, random_state=42)
        else:
            split_point = int(len(data) * split_ratio)
            data_train, data_test = data[:split_point], data[split_point:]
    return data_train, data_test


def _convert_dtypes(data_frame: pd.DataFrame):
    objects: pd.DataFrame = data_frame.select_dtypes('object')
    for column_name in objects:
        warnings.warn(f'Automatic factorization for the column {column_name} with type "object" is applied.')
        encoded = pd.factorize(data_frame[column_name])[0]
        data_frame[column_name] = encoded
    data_frame = data_frame.fillna(0)
    return data_frame


def train_test_data_setup(data: InputData, split_ratio=0.8,
                          shuffle_flag=False, task: Task = None) -> Tuple[InputData, InputData]:
    if data.features is not None:
        train_data_x, test_data_x = split_train_test(data.features, split_ratio, with_shuffle=shuffle_flag, task=task)
    else:
        train_data_x, test_data_x = None, None

    train_data_y, test_data_y = split_train_test(data.target, split_ratio, with_shuffle=shuffle_flag, task=task)
    train_idx, test_idx = split_train_test(data.idx, split_ratio, with_shuffle=shuffle_flag, task=task)
    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=train_idx, task=data.task, data_type=data.data_type)
    test_data = InputData(features=test_data_x, target=test_data_y, idx=test_idx, task=data.task,
                          data_type=data.data_type)
    return train_data, test_data


def _check_size_equality(outputs: List[OutputData]):
    """ Function check the size of combining datasets """
    idx_lengths = []
    idx_list = []
    for elem in outputs:
        idx_lengths.append(len(elem.idx))
        idx_list.append(elem.idx)

    # Check amount of unique lengths of datasets
    if len(set(idx_lengths)) == 1:
        are_lengths_equal = True
    else:
        are_lengths_equal = False

    return are_lengths_equal, idx_list


def merge_equal_outputs(outputs: List[OutputData]):
    """ Function merge datasets with equal amount of rows """

    features = []
    for elem in outputs:
        if len(elem.predict.shape) == 1:
            features.append(elem.predict)
        else:
            # if the model prediction is multivariate
            number_of_variables_in_prediction = elem.predict.shape[1]
            for i in range(number_of_variables_in_prediction):
                features.append(elem.predict[:, i])

    features = np.array(features).T
    idx = outputs[0].idx
    target = outputs[0].target
    return idx, features, target


def merge_non_equal_outputs(outputs: List[OutputData], idx_list: List):
    """ Function merge datasets with different amount of rows by idx field """
    # TODO add ability to merge datasets with different amount of features

    # Search overlapping indices in data
    for i, idx in enumerate(idx_list):
        idx = set(idx)
        if i == 0:
            common_idx = idx
        else:
            common_idx = common_idx & idx

    # Convert to list
    common_idx = np.array(list(common_idx))
    if len(common_idx) == 0:
        raise ValueError(f'There are no common indices for outputs')

    features = []

    for elem in outputs:
        # Create mask where True - appropriate objects
        mask = np.in1d(np.array(elem.idx), common_idx)

        if len(elem.predict.shape) == 1:
            filtered_predict = elem.predict[mask]
            features.append(filtered_predict)
        else:
            # if the model prediction is multivariate
            number_of_variables_in_prediction = elem.predict.shape[1]
            for i in range(number_of_variables_in_prediction):
                predict = elem.predict[:, i]
                filtered_predict = predict[mask]
                features.append(filtered_predict)

    old_target = outputs[-1].target
    filtered_target = old_target[mask]
    features = np.array(features).T
    return common_idx, features, filtered_target


def _combine_datasets_table(outputs: List[OutputData]):
    """ Function for combining datasets from parents to make features to
    another node. Features are tabular data.

    :param outputs: list with outputs from parent nodes
    :return idx: updated indices
    :return features: new features obtained from predictions at previous level
    :return target: updated target
    """

    are_lengths_equal, idx_list = _check_size_equality(outputs)

    if are_lengths_equal:
        idx, features, target = merge_equal_outputs(outputs)
    else:
        idx, features, target = merge_non_equal_outputs(outputs, idx_list)

    return idx, features, target


def _combine_datasets_ts(outputs: List[OutputData]):
    """ Function for combining datasets from parents to make features to
    another node. Features are time series data.

    :param outputs: list with outputs from parent nodes
    :return idx: updated indices
    :return features: new features obtained from predictions at previous level
    :return target: updated target
    """

    are_lengths_equal, idx_list = _check_size_equality(outputs)

    if are_lengths_equal:
        idx, features, target = merge_equal_outputs(outputs)
    else:
        idx, features, target = merge_non_equal_outputs(outputs, idx_list)

    features = np.ravel(np.array(features))
    target = np.ravel(np.array(target))
    return idx, features, target


def _combine_datasets_common(outputs: List[OutputData]):

    features = list()
    for elem in outputs:
        if len(elem.predict) != len(outputs[0].predict):
            raise NotImplementedError(f'Non-equal prediction length: '
                                      f'{len(elem.predict)} and {len(outputs[0].predict)}')
        if len(elem.predict.shape) == 1:
            features.append(elem.predict)
        else:
            # if the model prediction is multivariate
            number_of_variables_in_prediction = elem.predict.shape[1]
            for i in range(number_of_variables_in_prediction):
                features.append(elem.predict[:, i])
    return features
