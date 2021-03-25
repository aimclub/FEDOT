import os
import warnings

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
    features_idx: Optional[np.array] = None

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
    Data type for data predicted in the node
    """
    predict: np.array = None
    target: Optional[np.array] = None


class DataMerger:
    """
    Class for merging data, when it comes from different nodes and there is a
    need to merge it into next level node

    :param outputs: list with outputs from parent nodes
    """

    def __init__(self, outputs: List[OutputData]):
        self.outputs = outputs

    def merge(self):
        """ Method automatically determine which merge function should be
        applied """
        merge_function_by_type = {DataTypesEnum.ts: self.combine_datasets_ts,
                                  DataTypesEnum.table: self.combine_datasets_table,
                                  DataTypesEnum.text: self.combine_datasets_table,}

        first_data_type = self.outputs[0].data_type
        output_data_types = []
        for output in self.outputs:
            output_data_types.append(output.data_type)

        # Check is all data types can be merged or not
        if len(set(output_data_types)) > 1:
            raise ValueError("There is no ability to merge different data types")

        # Define appropriate strategy
        merge_func = merge_function_by_type.get(first_data_type)
        if merge_func is None:
            message = f"For data type '{first_data_type}' doesn't exist merge function"
            raise NotImplementedError(message)
        else:
            idx, features, target = merge_func()

        return idx, features, target

    def combine_datasets_table(self):
        """ Function for combining datasets from parents to make features to
        another node. Features are tabular data.

        :return idx: updated indices
        :return features: new features obtained from predictions at previous level
        :return target: updated target
        """
        are_lengths_equal, idx_list = self._check_size_equality(self.outputs)

        if are_lengths_equal:
            idx, features, target = self._merge_equal_outputs(self.outputs)
        else:
            idx, features, target = self._merge_non_equal_outputs(self.outputs,
                                                                  idx_list)

        return idx, features, target

    def combine_datasets_ts(self):
        """ Function for combining datasets from parents to make features to
        another node. Features are time series data.

        :return idx: updated indices
        :return features: new features obtained from predictions at previous level
        :return target: updated target
        """
        are_lengths_equal, idx_list = self._check_size_equality(self.outputs)

        if are_lengths_equal:
            idx, features, target = self._merge_equal_outputs(self.outputs)
        else:
            idx, features, target = self._merge_non_equal_outputs(self.outputs,
                                                                  idx_list)

        features = np.ravel(np.array(features))
        target = np.ravel(np.array(target))
        return idx, features, target

    @staticmethod
    def _merge_equal_outputs(outputs: List[OutputData]):
        """ Method merge datasets with equal amount of rows """

        features = []
        for elem in outputs:
            if len(elem.predict.shape) == 1:
                features.append(elem.predict)
            else:
                # If the model prediction is multivariate
                number_of_variables_in_prediction = elem.predict.shape[1]
                for i in range(number_of_variables_in_prediction):
                    features.append(elem.predict[:, i])

        features = np.array(features).T
        idx = outputs[0].idx
        target = outputs[0].target
        return idx, features, target

    @staticmethod
    def _merge_non_equal_outputs(outputs: List[OutputData], idx_list: List):
        """ Method merge datasets with different amount of rows by idx field """
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

    @staticmethod
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


def split_time_series(data, task):
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

    # Prepare data to train the model
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


def split_table(data, task, split_ratio, with_shuffle=False):
    """ Split table data into train and test parts

    :param data: array with data to split (not InputData)
    :param task: task to solve
    :param split_ratio: threshold for partitioning
    :param with_shuffle: is data needed to be shuffled or not
    """

    assert 0. <= split_ratio <= 1.
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

    # Prepare data to train the model
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
            train_data, test_data = split_time_series(data, task)
        elif data.data_type == DataTypesEnum.table:
            train_data, test_data = split_table(data, task, split_ratio,
                                                with_shuffle=shuffle_flag)
        else:
            train_data, test_data = split_table(data, task, split_ratio,
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
