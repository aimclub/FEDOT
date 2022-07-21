from __future__ import annotations
from functools import partial
from typing import List, Optional, Union

import os
import numpy as np
import pandas as pd

from fedot.core.data.data import process_target_and_features, get_indices_from_file, InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


class MultiModalData(dict):
    """ Dictionary with InputData as values and primary node names as keys """

    def __init__(self, *arg, **kw):
        super(MultiModalData, self).__init__(*arg, **kw)

        # Check if input data contains different targets
        self.contain_side_inputs = not all(value.supplementary_data.is_main_target for value in self.values())

    @property
    def idx(self):
        for input_data in self.values():
            if input_data.supplementary_data.is_main_target:
                return input_data.idx

    @property
    def task(self):
        for input_data in self.values():
            if input_data.supplementary_data.is_main_target:
                return input_data.task

    @task.setter
    def task(self, value):
        """ Update task for all input data """
        for input_data in self.values():
            input_data.task = value

    @property
    def target(self):
        """ Return main target from InputData blocks """
        for input_data in self.values():
            if input_data.supplementary_data.is_main_target:
                return input_data.target

    @target.setter
    def target(self, value):
        """ Update target for all input data """
        for input_data in self.values():
            input_data.target = value

    @property
    def data_type(self):
        return [input_data.data_type for input_data in iter(self.values())]

    @property
    def num_classes(self) -> Optional[int]:
        if self.task.task_type == TaskTypesEnum.classification:
            return len(np.unique(self.target))
        else:
            return None

    def shuffle(self):
        # TODO implement multi-modal shuffle
        pass

    def extract_data_source(self, source_name):
        """
            Function for extraction data_source from MultiModalData
            :param source_name: string with user-specified name of source
            :return target_data: selected source InputData
        """
        full_target_name = [key for key, _ in self.items() if source_name == key.split('/')[-1]][0]
        source_data = self[full_target_name]
        return source_data

    def subset_range(self, start: int, end: int):
        for key in self.keys():
            self[key] = self[key].subset_range(start, end)
        return self

    def subset_indices(self, selected_idx: List):
        for key in self.keys():
            self[key] = self[key].subset_indices(selected_idx)
        return self

    @classmethod
    def from_csv_time_series(cls,
                             task: Task,
                             file_path=None,
                             delimiter=',',
                             is_predict=False,
                             var_names=None,
                             target_column: Optional[str] = '') -> MultiModalData:
        df = pd.read_csv(file_path, sep=delimiter)
        idx = get_indices_from_file(df, file_path)

        if not var_names:
            var_names = list(set(df.columns) - set('datetime'))

        if is_predict:
            raise NotImplementedError(
                'Multivariate predict not supported in this function yet.')
        else:
            train_data, _ = \
                _prepare_multimodal_ts_data(dataframe=df,
                                            features=var_names,
                                            forecast_length=0)

            if target_column is not None:
                target = np.array(df[target_column])
            else:
                target = np.array(df[df.columns[-1]])

            # create labels for data sources
            data_part_transformation_func = partial(_array_to_input_data, idx=idx,
                                                    target_array=target, task=task,
                                                    data_type=DataTypesEnum.ts)

            sources = dict((_new_key_name_ts(data_part_key),
                            data_part_transformation_func(features_array=data_part))
                           for (data_part_key, data_part) in train_data.items())
            input_data = MultiModalData(sources)

        return input_data

    @classmethod
    def from_csv(cls,
                 file_path: Optional[Union[os.PathLike, str]] = None,
                 delimiter=',',
                 task: Task = Task(TaskTypesEnum.classification),
                 text_columns: Optional[Union[str, List[str]]] = None,
                 columns_to_drop: Optional[List[str]] = None,
                 target_columns: Union[str, List[str]] = '',
                 index_col: Optional[Union[str, int]] = 0) -> MultiModalData:
        """
        :param file_path: the path to the CSV with data
        :param columns_to_drop: the names of columns that should be dropped
        :param delimiter: the delimiter to separate the columns
        :param task: the task that should be solved with data
        :param text_columns: names of columns that contain text data
        :param target_columns: name of target column (last column if empty and no target if None)
        :param index_col: column name or index to use as the Data.idx;
            if None then arrange new unique index
        :return: MultiModalData object with text and table data sources as InputData
        """

        data_frame = pd.read_csv(file_path, sep=delimiter, index_col=index_col)
        if columns_to_drop:
            data_frame = data_frame.drop(columns_to_drop, axis=1)

        idx = data_frame.index.to_numpy()
        text_columns = [text_columns] if isinstance(text_columns, str) else text_columns

        if not text_columns:
            text_columns = _define_text_columns(data_frame)

        data_text = _prepare_multimodal_text_data(data_frame, text_columns)
        data_frame_table = data_frame.drop(columns=text_columns)
        table_features, target = process_target_and_features(data_frame_table, target_columns)

        data_part_transformation_func = partial(_array_to_input_data, idx=idx,
                                                target_array=target, task=task)

        # create labels for text data sources
        sources = dict((_new_key_name_text(data_part_key),
                        data_part_transformation_func(features_array=data_part, data_type=DataTypesEnum.text))
                       for (data_part_key, data_part) in data_text.items())

        # add table features if they exist
        if table_features.size != 0:
            sources.update({'data_source_table': data_part_transformation_func
                            (features_array=table_features, data_type=DataTypesEnum.table)})

        multi_modal_data = MultiModalData(sources)

        return multi_modal_data


def _define_text_columns(data_frame: pd.DataFrame) -> List[str]:
    """
    :param data_frame: pandas dataframe with data
    :return: list of text columns' names
    """
    text_columns = []
    for column_name in data_frame.columns:
        if _column_contains_text(data_frame[column_name]):
            text_columns.append(column_name)
    return text_columns


def _column_contains_text(column: pd.Series) -> bool:
    """
    Column contains text if:
    1. it's not numerical or latent numerical
    (e.g. ['1.2', '2.3', '3.4', ...] is numerical too)
    2. fraction of unique values is more than 0.95

    :param column: pandas series with data
    :return: True if column contains text
    """
    if column.dtype == object and not _is_float_compatible(column):
        return len(column.unique()) / len(column) > 0.95
    return False


def _is_float_compatible(column: pd.Series) -> bool:
    """
    :param column: pandas series with data
    :return: True if column contains only float or nan values
    """
    try:
        column.astype(float)
        return True
    except ValueError:
        return False


def _prepare_multimodal_text_data(dataframe: pd.DataFrame, text_columns: List[str]) -> dict:
    """ Prepares MultiModal text data in a form of dictionary

    :param dataframe: pandas DataFrame to process
    :param text_columns: list of text columns' names

    :return multimodal_text_data: dictionary with numpy arrays of text data
    """
    multi_modal_text_data = {}

    for column_name in text_columns:
        text_feature = np.array(dataframe[column_name])
        multi_modal_text_data.update({column_name: text_feature})

    return multi_modal_text_data


def _prepare_multimodal_ts_data(dataframe: pd.DataFrame, features: list, forecast_length: int) -> dict:
    """ Prepare MultiModal data for time series forecasting task in a form of
    dictionary

    :param dataframe: pandas DataFrame to process
    :param features: columns, which should be used as features in forecasting
    :param forecast_length: length of forecast

    :return multi_modal_train: dictionary with numpy arrays for train
    :return multi_modal_test: dictionary with numpy arrays for test
    """
    multi_modal_train = {}
    multi_modal_test = {}
    for feature in features:
        if forecast_length > 0:
            feature_ts = np.array(dataframe[feature])[:-forecast_length]
            idx = list(dataframe['datetime'])[:-forecast_length]
        else:
            feature_ts = np.array(dataframe[feature])
            idx = list(dataframe['datetime'])

        # Will be the same
        multi_modal_train.update({feature: feature_ts})
        multi_modal_test.update({feature: feature_ts})

    multi_modal_test['idx'] = np.asarray(idx)
    multi_modal_train['idx'] = np.asarray(idx)

    return multi_modal_train, multi_modal_test


def _array_to_input_data(features_array: np.array,
                         target_array: np.array,
                         idx: Optional[np.array] = None,
                         task: Task = Task(TaskTypesEnum.classification),
                         data_type: DataTypesEnum = DataTypesEnum.table) -> InputData:
    """
    Transforms numpy array to InputData object
    """
    if idx is None:
        idx = np.arange(len(features_array))

    return InputData(idx=idx, features=features_array, target=target_array, task=task, data_type=data_type)


def _new_key_name_ts(data_part_key: str) -> str:
    if data_part_key == 'idx':
        return 'idx'
    return f'data_source_ts/{data_part_key}'


def _new_key_name_text(data_part_key: str) -> str:
    return f'data_source_text/{data_part_key}'
