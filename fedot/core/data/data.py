from __future__ import annotations

import glob
import os
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Any
from collections.abc import Iterable

import numpy as np
import pandas as pd
from golem.core.log import default_log
from golem.utilities.requirements_notificator import warn_requirement

try:
    import cv2
except ModuleNotFoundError:
    warn_requirement('opencv-python', 'fedot[extra]')
    cv2 = None

from fedot.core.data.array_utilities import atleast_2d
from fedot.core.data.load_data import JSONBatchLoader, TextBatchLoader
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

#: The list of keyword for auto-detecting csv *tabular* data index. Used in :py:meth:`Data.from_csv`
#: and :py:meth:`MultiModalData.from_csv`.
POSSIBLE_TABULAR_IDX_KEYWORDS = ['idx', 'index', 'id', 'unnamed: 0']
#: The list of keyword for auto-detecting csv *time-series* data index. Used in :py:meth:`Data.from_csv_time_series`,
#: :py:meth:`Data.from_csv_multi_time_series` and :py:meth:`MultiModalData.from_csv_time_series`.
POSSIBLE_TS_IDX_KEYWORDS = ['datetime', 'date', 'time', 'unnamed: 0']

PathType = Union[os.PathLike, str]


@dataclass
class Data:
    """
    Base Data type class
    """

    idx: np.ndarray
    task: Task
    data_type: DataTypesEnum
    features: np.ndarray
    target: Optional[np.ndarray] = None

    # Object with supplementary info
    supplementary_data: SupplementaryData = field(default_factory=SupplementaryData)

    @classmethod
    def from_csv(cls,
                 file_path: PathType,
                 delimiter: str = ',',
                 task: Union[Task, str] = 'classification',
                 data_type: DataTypesEnum = DataTypesEnum.table,
                 columns_to_drop: Optional[List[Union[str, int]]] = None,
                 target_columns: Union[str, List[Union[str, int]]] = '',
                 index_col: Optional[Union[str, int]] = None,
                 possible_idx_keywords: Optional[List[str]] = None) -> InputData:
        """Import data from ``csv``.

        Args:
            file_path: the path to the ``CSV`` with data.
            columns_to_drop: the names of columns that should be dropped.
            delimiter: the delimiter to separate the columns.
            task: the :obj:`Task` to solve with the data.
            data_type: the type of the data. Possible values are listed at :class:`DataTypesEnum`.
            target_columns: name of the target column (the last column if empty and no target if ``None``).
            index_col: name or index of the column to use as the :obj:`Data.idx`.\n
                If ``None``, then check the first column's name and use it as index if succeeded
                (see the param ``possible_idx_keywords``).\n
                Set ``False`` to skip the check and rearrange a new integer index.
            possible_idx_keywords: lowercase keys to find. If the first data column contains one of the keys,
                it is used as index. See the :const:`POSSIBLE_TABULAR_IDX_KEYWORDS` for the list of default keywords.

        Returns:
            data
        """
        possible_idx_keywords = possible_idx_keywords or POSSIBLE_TABULAR_IDX_KEYWORDS
        if isinstance(task, str):
            task = Task(TaskTypesEnum(task))

        df = get_df_from_csv(file_path, delimiter, index_col, possible_idx_keywords, columns_to_drop=columns_to_drop)
        idx = df.index.to_numpy()

        features, target = process_target_and_features(df, target_columns)

        return InputData(idx=idx, features=features, target=target, task=task, data_type=data_type)

    @classmethod
    def from_csv_time_series(cls,
                             file_path: PathType,
                             delimiter: str = ',',
                             task: Union[Task, str] = 'ts_forecasting',
                             is_predict: bool = False,
                             columns_to_drop: Optional[List] = None,
                             target_column: Optional[str] = '',
                             index_col: Optional[Union[str, int]] = None,
                             possible_idx_keywords: Optional[List[str]] = None) -> InputData:
        """
        Forms :obj:`InputData` of ``ts`` type from columns of different variant of the same variable.

        Args:
            file_path: path to the source csv file.
            delimiter: delimiter for pandas DataFrame.
            task: the :obj:`Task` that should be solved with data.
            is_predict: indicator of stage to prepare the data to. ``False`` means fit, ``True`` means predict.
            columns_to_drop: ``list`` with names of columns to ignore.
            target_column: ``string`` with name of target column, used for predict stage.
            index_col: name or index of the column to use as the :obj:`Data.idx`.\n
                If ``None``, then check the first column's name and use it as index if succeeded
                (see the param ``possible_idx_keywords``).\n
                Set ``False`` to skip the check and rearrange a new integer index.
            possible_idx_keywords: lowercase keys to find. If the first data column contains one of the keys,
                it is used as index. See the :const:`POSSIBLE_TS_IDX_KEYWORDS` for the list of default keywords.

        Returns:
            An instance of :class:`InputData`.
        """

        possible_idx_keywords = possible_idx_keywords or POSSIBLE_TS_IDX_KEYWORDS
        if isinstance(task, str):
            task = Task(TaskTypesEnum(task))

        df = get_df_from_csv(file_path, delimiter, index_col, possible_idx_keywords, columns_to_drop=columns_to_drop)
        idx = df.index.to_numpy()

        if target_column is not None:
            time_series = np.array(df[target_column])
        else:
            time_series = np.array(df[df.columns[-1]])

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
            # Prepare InputData for train the pipeline
            input_data = InputData(idx=idx,
                                   features=time_series,
                                   target=time_series,
                                   task=task,
                                   data_type=DataTypesEnum.ts)

        return input_data

    @classmethod
    def from_csv_multi_time_series(cls,
                                   file_path: PathType,
                                   delimiter: str = ',',
                                   task: Union[Task, str] = 'ts_forecasting',
                                   is_predict: bool = False,
                                   columns_to_use: Optional[list] = None,
                                   target_column: Optional[str] = '',
                                   index_col: Optional[Union[str, int]] = None,
                                   possible_idx_keywords: Optional[List[str]] = None) -> InputData:
        """
        Forms :obj:`InputData` of ``multi_ts`` type from columns of different variant of the same variable

        Args:
            file_path: path to csv file.
            delimiter: delimiter for pandas df.
            task: the :obj:`Task` that should be solved with data.
            is_predict: indicator of stage to prepare the data to. ``False`` means fit, ``True`` means predict.
            columns_to_use: ``list`` with names of columns of different variant of the same variable.
            target_column: ``string`` with name of target column, used for predict stage.
            index_col: name or index of the column to use as the :obj:`Data.idx`.\n
                If ``None``, then check the first column's name and use it as index if succeeded
                (see the param ``possible_idx_keywords``).\n
                Set ``False`` to skip the check and rearrange a new integer index.
            possible_idx_keywords: lowercase keys to find. If the first data column contains one of the keys,
                it is used as index. See the :const:`POSSIBLE_TS_IDX_KEYWORDS` for the list of default keywords.

        Returns:
            An instance of :class:`InputData`.
        """

        df = get_df_from_csv(file_path, delimiter, index_col, possible_idx_keywords, columns_to_use=columns_to_use)
        idx = df.index.to_numpy()
        if columns_to_use is not None:
            actual_df = df[columns_to_use]
            multi_time_series = actual_df.to_numpy()
        else:
            multi_time_series = df.to_numpy()

        if is_predict:
            # Prepare data for prediction
            len_forecast = task.task_params.forecast_length
            if target_column is not None:
                time_series = np.array(df[target_column])
            else:
                time_series = np.array(df[df.columns[-1]])
            start_forecast = multi_time_series.shape[0]
            end_forecast = start_forecast + len_forecast
            input_data = InputData(idx=np.arange(start_forecast, end_forecast),
                                   features=time_series,
                                   target=None,
                                   task=task,
                                   data_type=DataTypesEnum.multi_ts)
        else:
            # Prepare InputData for train the pipeline
            input_data = InputData(idx=idx,
                                   features=multi_time_series,
                                   target=multi_time_series,
                                   task=task,
                                   data_type=DataTypesEnum.multi_ts)

        return input_data

    @staticmethod
    def from_image(images: Union[str, np.ndarray] = None,
                   labels: Union[str, np.ndarray] = None,
                   task: Task = Task(TaskTypesEnum.classification),
                   target_size: Optional[Tuple[int, int]] = None) -> InputData:
        """Input data from Image

        Args:
            images: the path to the directory with image data in ``np.ndarray`` format
                or array in ``np.ndarray`` format
            labels: the path to the directory with image labels in ``np.ndarray`` format
                or array in ``np.ndarray`` format
            task: the :obj:`Task` that should be solved with data
            target_size: size for the images resizing (if necessary)

        Returns:
            An instance of :class:`InputData`.
        """

        features = images
        target = labels

        if type(images) is str:
            # if upload from path
            if '*.jpeg' in images:
                # upload from folder of images
                path = images
                images_list = []
                for file_path in glob.glob(path):
                    if target_size is not None:
                        img = _resize_image(file_path, target_size)
                        images_list.append(img)
                    else:
                        raise ValueError('Set target_size for images')
                features = np.asarray(images_list)
                target = labels
            else:
                # upload from array
                features = np.load(images)
                target = np.load(labels)
                # add channels if None
                if len(features.shape) == 3:
                    features = np.expand_dims(features, -1)

        idx = np.arange(0, len(features))

        return InputData(idx=idx, features=features, target=target, task=task, data_type=DataTypesEnum.image)

    @staticmethod
    def from_text_meta_file(meta_file_path: str = None,
                            label: str = 'label',
                            task: Task = Task(TaskTypesEnum.classification),
                            data_type: DataTypesEnum = DataTypesEnum.text) -> InputData:

        if os.path.isdir(meta_file_path):
            raise ValueError("""CSV file expected but got directory""")

        df_text = pd.read_csv(meta_file_path)
        df_text = df_text.sample(frac=1).reset_index(drop=True)
        messages = df_text['text'].astype('U').tolist()

        features = np.array(messages)
        target = np.array(df_text[label]).reshape(-1, 1)
        idx = [index for index in range(len(target))]

        return InputData(idx=idx, features=features,
                         target=target, task=task, data_type=data_type)

    @staticmethod
    def from_text_files(files_path: str,
                        label: str = 'label',
                        task: Task = Task(TaskTypesEnum.classification),
                        data_type: DataTypesEnum = DataTypesEnum.text) -> InputData:

        if os.path.isfile(files_path):
            raise ValueError("""Path to the directory expected but got file""")

        df_text = TextBatchLoader(path=files_path).extract()

        features = np.array(df_text['text'])
        target = np.array(df_text[label]).reshape(-1, 1)
        idx = [index for index in range(len(target))]

        return InputData(idx=idx, features=features,
                         target=target, task=task, data_type=data_type)

    @staticmethod
    def from_json_files(files_path: str,
                        fields_to_use: List,
                        label: str = 'label',
                        task: Task = Task(TaskTypesEnum.classification),
                        data_type: DataTypesEnum = DataTypesEnum.table,
                        export_to_meta=False, is_multilabel=False, shuffle=True) -> InputData:
        """Generates InputData from the set of ``JSON`` files with different fields

        Args:
            files_path: path the folder with ``json`` files
            fields_to_use: ``list`` of fields that will be considered as a features
            label: name of field with target variable
            task: :obj:`Task` to solve
            data_type: data type in fields (as well as type for obtained :obj:`InputData`)
            export_to_meta: combine extracted field and save to ``CSV``
            is_multilabel: if ``True``, creates multilabel target
            shuffle: if ``True``, shuffles data

        Returns:
            An instance of :class:`InputData`.
        """

        if os.path.isfile(files_path):
            raise ValueError("""Path to the directory expected but got file""")

        df_data = JSONBatchLoader(path=files_path, label=label, fields_to_use=fields_to_use,
                                  shuffle=shuffle).extract(export_to_meta)

        if len(fields_to_use) > 1:
            fields_to_combine = []
            for field_to_use in fields_to_use:
                fields_to_combine.append(np.array(df_data[field_to_use]))
                # Unite if the element of text data is divided into strings
                if isinstance(df_data[field_to_use][0], list):
                    df_data[field_to_use] = [' '.join(piece) for piece in df_data[field_to_use]]

            features = np.column_stack(tuple(fields_to_combine))
        else:
            field_to_use = df_data[fields_to_use[0]]
            # process field_to_use with nested list
            if isinstance(field_to_use[0], list):
                field_to_use = [' '.join(piece) for piece in field_to_use]
            features = np.array(field_to_use)

        if is_multilabel:
            target = df_data[label]
            classes = set()
            for el in target:
                for label in el:
                    classes.add(label)
            count_classes = list(sorted(classes))
            multilabel_target = np.zeros((len(features), len(count_classes)))

            for i in range(len(target)):
                for el in target[i]:
                    multilabel_target[i][count_classes.index(el)] = 1
            target = multilabel_target
        else:
            target = np.array(df_data[label])

        idx = [index for index in range(len(target))]

        return InputData(idx=idx, features=features,
                         target=target, task=task, data_type=data_type)

    def to_csv(self, path_to_save):
        dataframe = pd.DataFrame(data=self.features, index=self.idx)
        if self.target is not None:
            dataframe['target'] = self.target
        dataframe.to_csv(path_to_save)


@dataclass
class InputData(Data):
    """Data class for input data for the nodes
    """

    @property
    def is_classification(self):
        return self.task.task_type is TaskTypesEnum.classification

    @property
    def is_ts_forecasting(self):
        return self.task.task_type is TaskTypesEnum.ts_forecasting

    @property
    def is_clustering(self):
        return self.task.task_type is TaskTypesEnum.clustering

    @property
    def is_regression(self):
        return self.task.task_type is TaskTypesEnum.regression

    @property
    def is_ts(self):
        return self.data_type is DataTypesEnum.ts

    @property
    def is_multi_ts(self):
        return self.data_type is DataTypesEnum.multi_ts

    @property
    def is_table(self):
        return self.data_type is DataTypesEnum.table

    @property
    def is_text(self):
        return self.data_type is DataTypesEnum.text

    @property
    def is_image(self):
        return self.data_type is DataTypesEnum.image

    @property
    def num_classes(self) -> Optional[int]:
        """Returns number of classes that are present in the target.
        NB: if some labels are not present in this data, then
        number of classes can be less than in the full dataset!"""
        unique_values = self.class_labels
        return len(unique_values) if unique_values is not None else None

    @property
    def class_labels(self) -> Optional[int]:
        """Returns unique class labels that are present in the target"""
        if self.task.task_type == TaskTypesEnum.classification and self.target is not None:
            return np.unique(self.target)
        else:
            return None

    def __len__(self):
        return len(self.idx)

    def slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = 1):
        if start is None and stop is None:
            raise ValueError('Slicing range is undefined')

        index = np.arange(len(self))[slice(start, stop, step)]

        return self.slice_by_index(index, step)

    def slice_by_index(self, indexes: Iterable, step: int = 1):
        """ Extract data with indexes (not ``idx``)
            Save features before first index in ``indexes`` for time series
            :param indexes: iterator with indexes that should be extracted
                            should be sorted for time series
            :param step: step between indexes for time series
            :return: InputData """
        new = self.copy()
        if self.task.task_type is TaskTypesEnum.ts_forecasting:
            # retain data in features before ``indexes``
            delta = new.features.shape[0] - len(new)
            new_indexes = np.arange(-delta, indexes[-1] + 1)[::-step][::-1]
            new_indexes += delta
            new.features = np.take(new.features, new_indexes, 0)
        else:
            new.features = np.take(new.features, indexes, 0)
        new.idx = np.take(new.idx, indexes, 0)
        new.target = np.take(new.target, indexes, 0)
        return new

    def subset_range(self, start: int, end: int):
        if start > end or not (0 <= start <= end <= len(self)):
            raise IndexError(f'Incorrect indexes in slice {slice} for data with length {len(self)}')
        return self.slice(start, end + 1)

    def subset_indices(self, selected_idx: List):
        """Get subset from :obj:`InputData` to extract all items with specified indices

        Args:
            selected_idx: ``list`` of indices for extraction

        Returns:
            :obj:`InputData`
        """

        hash_table = {str(idx): num for num, idx in enumerate(self.idx)}
        try:
            row_nums = [hash_table[str(selected_ind)] for selected_ind in selected_idx]
        except KeyError:
            missing_values = [selected_ind for selected_ind in selected_idx if selected_ind not in hash_table]
            raise IndexError(f"Next indexes are missing: {missing_values}")
        return self.slice_by_index(row_nums)

    def subset_features(self, features_ids: list, with_target: bool = False):
        """Return new :obj:`InputData` with subset of features based on ``features_ids`` list
        """
        subsample_input = self.copy()
        subsample_input.features = self.features[:, features_ids]
        if with_target:
            if self.target.shape[1] != self.features.shape[1]:
                raise ValueError((f"Shapes of features ({self.features.shape}) and"
                                 f" target ({self.target.shape}) mismatch. Cannot create subset for target"))
            subsample_input.target = self.target[:, features_ids]
        return subsample_input

    def shuffle(self, seed: Optional[int] = None):
        """Shuffles features and target if possible
        """

        if self.data_type in (DataTypesEnum.table, DataTypesEnum.image, DataTypesEnum.text):
            if seed is None:
                seed = np.random.randint(0, np.iinfo(int).max)
            generator = np.random.RandomState(seed)
            shuffled_ind = generator.permutation(len(self))

            self.features = np.take(self.features, shuffled_ind, 0)
            self.idx = np.take(self.idx, shuffled_ind, 0)
            self.target = np.take(self.target, shuffled_ind, 0)

    def convert_non_int_indexes_for_fit(self, pipeline):
        """Conversion non ``int`` (``datetime``, ``string``, etc) indexes in ``integer`` form on the fit stage
        """

        copied_data = deepcopy(self)
        is_timestamp = isinstance(copied_data.idx[0], pd._libs.tslibs.timestamps.Timestamp)
        is_numpy_datetime = isinstance(copied_data.idx[0], np.datetime64)
        # if fit stage- just creating range of integers
        if is_timestamp or is_numpy_datetime:
            copied_data.supplementary_data.non_int_idx = copy(copied_data.idx)
            copied_data.idx = np.array(range(len(copied_data.idx)))

            last_idx_time = copied_data.supplementary_data.non_int_idx[-1]
            pre_last_time = copied_data.supplementary_data.non_int_idx[-2]

            pipeline.last_idx_int = copied_data.idx[-1]
            pipeline.last_idx_dt = last_idx_time
            pipeline.period = last_idx_time - pre_last_time
        elif not isinstance(copied_data.idx[0], (int, np.int32, np.int64)):
            copied_data.supplementary_data.non_int_idx = copy(copied_data.idx)
            copied_data.idx = np.array(range(len(copied_data.idx)))
            pipeline.last_idx_int = copied_data.idx[-1]
        return copied_data

    def convert_non_int_indexes_for_predict(self, pipeline):
        """Conversion non ``int`` (``datetime``, ``string``, etc) indexes in ``integer`` form on the predict stage
        """

        copied_data = deepcopy(self)
        is_timestamp = isinstance(copied_data.idx[0], pd._libs.tslibs.timestamps.Timestamp)
        is_numpy_datetime = isinstance(copied_data.idx[0], np.datetime64)
        # if predict stage - calculating shift from last train part index
        if is_timestamp or is_numpy_datetime:
            copied_data.supplementary_data.non_int_idx = copy(self.idx)
            copied_data.idx = self._resolve_non_int_idx(pipeline)
        elif not isinstance(copied_data.idx[0], (int, np.int32, np.int64)):
            # note, that string indexes do not have an order and always we think that indexes we want to predict go
            # immediately after the train indexes
            copied_data.supplementary_data.non_int_idx = copy(copied_data.idx)
            copied_data.idx = pipeline.last_idx_int + np.array(range(1, len(copied_data.idx) + 1))
        return copied_data

    def copy(self):
        return InputData(idx=self.idx,
                         features=self.features,
                         target=self.target,
                         task=self.task,
                         data_type=self.data_type,
                         supplementary_data=deepcopy(self.supplementary_data))


    @staticmethod
    def _resolve_func(pipeline, x):
        return pipeline.last_idx_int + (x - pipeline.last_idx_dt) // pipeline.period

    def _resolve_non_int_idx(self, pipeline):
        return np.array(list(map(lambda x: self._resolve_func(pipeline, x), self.idx)))


@dataclass
class OutputData(Data):
    """``Data`` type for data prediction in the node
    """

    features: Optional[np.ndarray] = None
    predict: np.ndarray = None
    target: Optional[np.ndarray] = None


def _resize_image(file_path: str, target_size: Tuple[int, int]):
    """Function resizes and rewrites the input image
    """

    img = cv2.imread(file_path)
    if img.shape[:2] != target_size:
        img = cv2.resize(img, (target_size[0], target_size[1]))
        cv2.imwrite(file_path, img)
    return img


def process_target_and_features(data_frame: pd.DataFrame,
                                target_column: Optional[Union[str, List[str]]]
                                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Function process pandas ``dataframe`` with single column

    Args:
        data_frame: loaded pandas :obj:`DataFrame`
        target_column: names of columns with target or ``None``

    Returns:
        (``np.array`` (table) with features, ``np.array`` (column) with target)
    """

    if target_column == '':
        # Take the last column in the table
        target_column = data_frame.columns[-1]

    if target_column:
        target = atleast_2d(data_frame[target_column].to_numpy())
        features = data_frame.drop(columns=target_column).to_numpy()
    else:
        target = None
        features = data_frame.to_numpy()

    return features, target


def get_indices_from_file(data_frame, file_path, idx_column='datetime') -> Iterable[Any]:
    if idx_column in data_frame.columns:
        df = pd.read_csv(file_path,
                         parse_dates=[idx_column])
        idx = [str(d) for d in df[idx_column]]
        return idx
    return np.arange(0, len(data_frame))


def np_datetime_to_numeric(data: np.ndarray) -> np.ndarray:
    """
    Change data's datetime type to integer with milliseconds unit.

    Args:
        data: table data for converting.

    Returns:
        The same table data with datetimes (if existed) converted to integer
    """
    orig_shape = data.shape
    out_dtype = np.int64 if 'datetime' in str((dt := data.dtype)) else dt
    features_df = pd.DataFrame(data, copy=False).infer_objects()
    date_cols = features_df.select_dtypes('datetime')
    converted_cols = date_cols.to_numpy(np.int64) // 1e6  # to 'ms' unit from 'ns'
    features_df[date_cols.columns] = converted_cols
    return features_df.to_numpy(out_dtype).reshape(orig_shape)


def array_to_input_data(features_array: np.ndarray,
                        target_array: np.ndarray,
                        idx: Optional[np.ndarray] = None,
                        task: Task = Task(TaskTypesEnum.classification),
                        data_type: Optional[DataTypesEnum] = None) -> InputData:
    if idx is None:
        idx = np.arange(len(features_array))
    if data_type is None:
        data_type = autodetect_data_type(task)
    return InputData(idx=idx, features=features_array, target=target_array, task=task, data_type=data_type)


def autodetect_data_type(task: Task) -> DataTypesEnum:
    if task.task_type == TaskTypesEnum.ts_forecasting:
        return DataTypesEnum.ts
    else:
        return DataTypesEnum.table


def get_df_from_csv(file_path: PathType, delimiter: str, index_col: Optional[Union[str, int]] = None,
                    possible_idx_keywords: Optional[List[str]] = None, *,
                    columns_to_drop: Optional[List[Union[str, int]]] = None,
                    columns_to_use: Optional[List[Union[str, int]]] = None):
    def define_index_column(candidate_columns: List[str]) -> Optional[str]:
        for column_name in candidate_columns:
            if is_column_name_suitable_for_index(column_name):
                return column_name

    def is_column_name_suitable_for_index(column_name: str) -> bool:
        return any(key in column_name.lower() for key in possible_idx_keywords)

    columns_to_drop = copy(columns_to_drop) or []
    columns_to_use = copy(columns_to_use) or []
    possible_idx_keywords = possible_idx_keywords or []

    logger = default_log('CSV data extraction')

    columns = pd.read_csv(file_path, sep=delimiter, index_col=False, nrows=1).columns

    if columns_to_drop and columns_to_use:
        raise ValueError('Incompatible arguments are used: columns_to_drop and columns_to_use. '
                         'Only one of them can be specified simultaneously.')

    if columns_to_drop:
        columns_to_use = [col for col in columns if col not in columns_to_drop]
    elif not columns_to_use:
        columns_to_use = list(columns)

    candidate_idx_cols = [columns_to_use[0], columns[0]]
    if index_col is None:
        defined_index = define_index_column(candidate_idx_cols)
        if defined_index is not None:
            index_col = defined_index
            logger.message(f'Used the column as index: "{index_col}".')

    if (index_col is not None) and (index_col not in columns_to_use):
        columns_to_use.append(index_col)

    return pd.read_csv(file_path, sep=delimiter, index_col=index_col, usecols=columns_to_use)
