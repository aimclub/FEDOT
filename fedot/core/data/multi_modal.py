from __future__ import annotations

from functools import partial
from typing import List, Optional, Union, Dict

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.data.data import (process_target_and_features, array_to_input_data,
                                  get_df_from_csv, PathType, POSSIBLE_TABULAR_IDX_KEYWORDS, POSSIBLE_TS_IDX_KEYWORDS)
from fedot.core.data.data_detection import TextDataDetector, TimeSeriesDataDetector
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


class MultiModalData(Dict[str, InputData]):
    """ Dictionary with InputData as values and primary node names as keys """

    def __init__(self, *arg, **kw):
        super().__init__(*arg, **kw)

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
        unique_values = self.class_labels
        return len(unique_values) if unique_values is not None else None

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
    def from_csv(cls,
                 file_path: Optional[PathType],
                 delimiter=',',
                 task: Union[Task, str] = 'classification',
                 text_columns: Optional[Union[str, List[str]]] = None,
                 columns_to_drop: Optional[List[str]] = None,
                 target_columns: Union[str, List[str]] = '',
                 index_col: Optional[Union[str, int]] = None,
                 possible_idx_keywords: Optional[List[str]] = None) -> MultiModalData:
        """Import multimodal data from ``csv``.

        Args:
            file_path: the path to the ``CSV`` with data.
            delimiter: the delimiter to separate the columns.
            task: the :obj:`Task` to solve with the data.
            text_columns: names of columns that contain text data.
            columns_to_drop: the names of columns that should be dropped.
            target_columns: name of the target column (the last column if empty and no target if ``None``).
            index_col: name or index of the column to use as the :obj:`Data.idx`.\n
                If ``None``, then check the first column's name and use it as index if succeeded
                (see the param ``possible_idx_keywords``).\n
                Set ``False`` to skip the check and rearrange a new integer index.
            possible_idx_keywords: lowercase keys to find. If the first data column contains one of the keys,
                it is used as index. See the :const:`POSSIBLE_TABULAR_IDX_KEYWORDS` for the list of default
                keywords.

        Returns:
            An instance of :class:`MultiModalData` containing text and table data sources as :class:`InputData`
                instances.
        """
        possible_idx_keywords = possible_idx_keywords or POSSIBLE_TABULAR_IDX_KEYWORDS
        data_frame = get_df_from_csv(file_path, delimiter, index_col, possible_idx_keywords,
                                     columns_to_drop=columns_to_drop)
        idx = data_frame.index.to_numpy()
        if isinstance(task, str):
            task = Task(TaskTypesEnum(task))

        text_columns = [text_columns] if isinstance(text_columns, str) else text_columns
        text_data_detector = TextDataDetector()
        if not text_columns:
            text_columns = text_data_detector.find_text_columns(data_frame)

        link_columns = text_data_detector.find_link_columns(data_frame)
        columns_to_drop = text_columns + link_columns
        data_text = text_data_detector.prepare_multimodal_data(data_frame, text_columns)
        data_frame_table = data_frame.drop(columns=columns_to_drop)
        table_features, target = process_target_and_features(data_frame_table, target_columns)

        data_part_transformation_func = partial(array_to_input_data,
                                                idx=idx, target_array=target, task=task)

        # create labels for text data sources and remove source if there are many nans or text is link
        sources = dict((text_data_detector.new_key_name(data_part_key),
                        data_part_transformation_func(features_array=data_part, data_type=DataTypesEnum.text))
                       for (data_part_key, data_part) in data_text.items()
                       if not text_data_detector.is_full_of_nans(data_part))

        # add table features if they exist
        if table_features.size != 0:
            sources.update({'data_source_table': data_part_transformation_func(features_array=table_features,
                                                                               data_type=DataTypesEnum.table)})

        multi_modal_data = MultiModalData(sources)

        return multi_modal_data

    @classmethod
    def from_csv_time_series(cls,
                             file_path: PathType,
                             delimiter: str = ',',
                             task: Union[Task, str] = 'ts_forecasting',
                             is_predict: bool = False,
                             columns_to_use: Optional[list] = None,
                             target_column: Optional[str] = '',
                             index_col: Optional[Union[str, int]] = None,
                             possible_idx_keywords: Optional[List[str]] = None) -> MultiModalData:
        """Import multimodal data from ``csv``.

        Args:
            file_path: the path to the ``CSV`` with data.
            delimiter: the delimiter to separate the columns.
            task: the :obj:`Task` to solve with the data.
            is_predict: indicator of stage to prepare the data to. ``False`` means fit, ``True`` means predict.
            columns_to_use: ``list`` with names of columns of different variant of the same variable.
            target_column: ``string`` with name of target column, used for predict stage.
            index_col: name or index of the column to use as the :obj:`Data.idx`.\n
                If ``None``, then check the first column's name and use it as index if succeeded
                (see the param ``possible_idx_keywords``).\n
                Set ``False`` to skip the check and rearrange a new integer index.
            possible_idx_keywords: lowercase keys to find. If the first data column contains one of the keys,
                it is used as index. See the :const:`POSSIBLE_TS_IDX_KEYWORDS` for the list of default
                keywords.

        Returns:
            An instance of :class:`MultiModalData` multiple time series data sources as :class:`InputData` instances.
        """

        possible_idx_keywords = possible_idx_keywords or POSSIBLE_TS_IDX_KEYWORDS
        if isinstance(task, str):
            task = Task(TaskTypesEnum(task))

        df = get_df_from_csv(file_path, delimiter, index_col, possible_idx_keywords, columns_to_use=columns_to_use)
        idx = df.index.to_numpy()
        if not columns_to_use:
            columns_to_use = list(set(df.columns) - {index_col})

        if is_predict:
            raise NotImplementedError(
                'Multivariate predict not supported in this function yet.')

        ts_data_detector = TimeSeriesDataDetector()
        data = ts_data_detector.prepare_multimodal_data(dataframe=df,
                                                        columns=columns_to_use)

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
