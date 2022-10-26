from __future__ import annotations

import os
from functools import partial
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from fedot.core.data.data import process_target_and_features, get_indices_from_file, array_to_input_data
from fedot.core.data.data_detection import TextDataDetector, TimeSeriesDataDetector
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

    @property
    def class_labels(self) -> Optional[List[Union[int, str, float]]]:
        if self.task.task_type == TaskTypesEnum.classification and self.target is not None:
            return np.unique(self.target)
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
                             task: Union[Task, str] = 'ts_forecasting',
                             file_path=None,
                             delimiter=',',
                             is_predict=False,
                             var_names=None,
                             target_column: Optional[str] = '',
                             idx_column: Optional[str] = 'datetime') -> MultiModalData:
        ts_data_detector = TimeSeriesDataDetector()
        df = pd.read_csv(file_path, sep=delimiter)
        idx = get_indices_from_file(df, file_path, idx_column)
        if isinstance(task, str):
            task = Task(TaskTypesEnum(task))
        if not var_names:
            var_names = list(set(df.columns) - set(idx_column))

        if is_predict:
            raise NotImplementedError(
                'Multivariate predict not supported in this function yet.')
        else:
            data = ts_data_detector.prepare_multimodal_data(dataframe=df,
                                                            columns=var_names)

            if target_column is not None:
                target = np.array(df[target_column])
            else:
                target = np.array(df[df.columns[-1]])

            # create labels for data sources
            data_part_transformation_func = partial(array_to_input_data,
                                                    idx=idx, target_array=target, task=task,
                                                    data_type=DataTypesEnum.ts)

            sources = dict((ts_data_detector.new_key_name(data_part_key),
                            data_part_transformation_func(features_array=data_part))
                           for (data_part_key, data_part) in data.items())
            multi_modal_data = MultiModalData(sources)

        return multi_modal_data

    @classmethod
    def from_csv(cls,
                 file_path: Optional[Union[os.PathLike, str]] = None,
                 delimiter=',',
                 task: Union[Task, str] = 'classification',
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

        text_data_detector = TextDataDetector()
        data_frame = pd.read_csv(file_path, sep=delimiter, index_col=index_col)
        if columns_to_drop:
            data_frame = data_frame.drop(columns_to_drop, axis=1)

        idx = data_frame.index.to_numpy()
        if isinstance(task, str):
            task = Task(TaskTypesEnum(task))
        text_columns = [text_columns] if isinstance(text_columns, str) else text_columns

        if not text_columns:
            text_columns = text_data_detector.define_text_columns(data_frame)

        data_text = text_data_detector.prepare_multimodal_data(data_frame, text_columns)
        data_frame_table = data_frame.drop(columns=text_columns)
        table_features, target = process_target_and_features(data_frame_table, target_columns)

        data_part_transformation_func = partial(array_to_input_data,
                                                idx=idx, target_array=target, task=task)

        # create labels for text data sources and remove source if there are many nans
        sources = dict((text_data_detector.new_key_name(data_part_key),
                        data_part_transformation_func(features_array=data_part, data_type=DataTypesEnum.text))
                       for (data_part_key, data_part) in data_text.items()
                       if not text_data_detector.is_full_of_nans(data_part))

        # add table features if they exist
        if table_features.size != 0:
            sources.update({'data_source_table': data_part_transformation_func
                            (features_array=table_features, data_type=DataTypesEnum.table)})

        multi_modal_data = MultiModalData(sources)

        return multi_modal_data
