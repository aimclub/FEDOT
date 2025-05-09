from __future__ import annotations

import glob
import os
from copy import copy, deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

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
    features: Union[np.ndarray, pd.DataFrame]
    categorical_features: Optional[np.ndarray] = None
    categorical_idx: Optional[np.ndarray] = None
    numerical_idx: Optional[np.ndarray] = None
    encoded_idx: Optional[np.ndarray] = None
    features_names: Optional[np.ndarray[str]] = None
    target: Optional[np.ndarray] = None

    # Object with supplementary info
    supplementary_data: SupplementaryData = field(default_factory=SupplementaryData)

    @classmethod
    def from_numpy(cls,
                   features_array: np.ndarray,
                   target_array: Optional[np.ndarray] = None,
                   idx: Optional[np.ndarray] = None,
                   task: Union[Task, str] = 'classification',
                   data_type: Optional[DataTypesEnum] = DataTypesEnum.table,
                   features_names: np.ndarray[str] = None,
                   categorical_idx: Union[list[int, str], np.ndarray[int, str]] = None) -> InputData:
        """Import data from numpy array.

        Args:
            features_array: numpy array with features.
            target_array: numpy array with target.
            features_names: numpy array with names of features
            categorical_idx: a list or numpy array with indexes or names of features (if provided feature_names)
                that indicate that the feature is categorical.
            idx: indices of arrays.
            task: the :obj:`Task` to solve with the data.
            data_type: the type of the data. Possible values are listed at :class:`DataTypesEnum`.

        Returns:
            data: :InputData: representation of data in an internal data structure.
        """
        if isinstance(task, str):
            task = Task(TaskTypesEnum(task))
        return array_to_input_data(features_array=features_array,
                                   target_array=target_array,
                                   features_names=features_names,
                                   categorical_idx=categorical_idx,
                                   idx=idx,
                                   task=task,
                                   data_type=data_type)

    @classmethod
    def from_numpy_time_series(cls,
                               features_array: np.ndarray,
                               target_array: Optional[np.ndarray] = None,
                               idx: Optional[np.ndarray] = None,
                               task: Union[Task, str] = 'ts_forecasting',
                               data_type: Optional[DataTypesEnum] = DataTypesEnum.ts) -> InputData:
        """Import time series from numpy array.

        Args:
            features_array: numpy array with features time series.
            target_array: numpy array with target time series (if None same as features).
            idx: indices of arrays.
            task: the :obj:`Task` to solve with the data.
            data_type: the type of the data. Possible values are listed at :class:`DataTypesEnum`.

        Returns:
            data: :InputData: representation of data in an internal data structure.
        """
        if isinstance(task, str):
            task = Task(TaskTypesEnum(task))
        if target_array is None:
            target_array = features_array
        return array_to_input_data(features_array=features_array,
                                   target_array=target_array,
                                   idx=idx,
                                   task=task,
                                   data_type=data_type)

    @classmethod
    def from_dataframe(cls,
                       features_df: Union[pd.DataFrame, pd.Series],
                       target_df: Optional[Union[pd.DataFrame, pd.Series]] = None,
                       categorical_idx: Union[list[int, str], np.ndarray[int, str]] = None,
                       task: Union[Task, str] = 'classification',
                       data_type: DataTypesEnum = DataTypesEnum.table) -> InputData:
        """Import data from pandas DataFrame.

        Args:
            features_df: loaded pandas DataFrame or Series with features.
            target_df: loaded pandas DataFrame or Series with target.
            categorical_idx: a list or numpy array with indexes or names of features that indicate that
                the feature is categorical.
            task: the :obj:`Task` to solve with the data.
            data_type: the type of the data. Possible values are listed at :class:`DataTypesEnum`.

        Returns:
            data: :InputData: representation of data in an internal data structure.
        """

        if isinstance(task, str):
            task = Task(TaskTypesEnum(task))
        if isinstance(features_df, pd.Series):
            features_df = pd.DataFrame(features_df)
        if isinstance(target_df, pd.Series):
            target_df = pd.DataFrame(target_df)

        idx = features_df.index.to_numpy()
        features_names = features_df.columns.to_numpy()

        if target_df is not None:
            target_columns = target_df.columns.to_list()
            df = pd.concat([features_df, target_df], axis=1)
            features, target = process_target_and_features(df, target_columns)
        else:
            features, target = process_target_and_features(features_df, target_column=None)

        categorical_features = None
        if categorical_idx is not None:
            if isinstance(categorical_idx, list):
                categorical_idx = np.array(categorical_idx)

            if categorical_idx.size != 0 and isinstance(categorical_idx[0], str) and features_names is None:
                raise ValueError(
                    'Impossible to specify categorical features by name when the features_names are not specified'
                )

            if categorical_idx.size != 0 and isinstance(categorical_idx[0], str):
                categorical_idx = np.array(
                    [idx for idx, column in enumerate(features_names) if column in set(categorical_idx)]
                )

            if categorical_idx.size != 0:
                categorical_features = features[:, categorical_idx]

        data = InputData(
            idx=idx,
            features=features,
            target=target,
            task=task,
            data_type=data_type,
            features_names=features_names,
            categorical_idx=categorical_idx,
            categorical_features=categorical_features
        )

        return data

    @classmethod
    def from_csv(cls,
                 file_path: PathType,
                 delimiter: str = ',',
                 task: Union[Task, str] = 'classification',
                 data_type: DataTypesEnum = DataTypesEnum.table,
                 columns_to_drop: Optional[List[Union[str, int]]] = None,
                 target_columns: Union[str, List[Union[str, int]], None] = '',
                 categorical_idx: Union[list[int, str], np.ndarray[int, str]] = None,
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
            categorical_idx: a list or numpy array with indexes or names of features that indicate that
                the feature is categorical.
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

        if target_columns:
            features_names = df.drop(target_columns, axis=1).columns.to_numpy()

        else:
            features_names = df.columns.to_numpy()

        features, target = process_target_and_features(df, target_columns)

        categorical_features = None
        if categorical_idx is not None:
            if isinstance(categorical_idx, list):
                categorical_idx = np.array(categorical_idx)

            if categorical_idx.size != 0 and isinstance(categorical_idx[0], str) and features_names is None:
                raise ValueError(
                    'Impossible to specify categorical features by name when the features_names are not specified'
                )

            if categorical_idx.size != 0 and isinstance(categorical_idx[0], str):
                categorical_idx = np.array(
                    [idx for idx, column in enumerate(features_names) if column in set(categorical_idx)]
                )

            if categorical_idx.size != 0:
                categorical_features = features[:, categorical_idx]

        data = InputData(
            idx=idx,
            features=features,
            target=target,
            task=task,
            data_type=data_type,
            features_names=features_names,
            categorical_idx=categorical_idx,
            categorical_features=categorical_features
        )

        return data

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

        if isinstance(images, str):
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
        idx = np.array([index for index in range(len(target))])

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
        idx = np.array([index for index in range(len(target))])

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

        idx = np.array([index for index in range(len(target))])

        return InputData(idx=idx, features=features,
                         target=target, task=task, data_type=data_type)

    def to_csv(self, path_to_save):
        dataframe = pd.DataFrame(data=self.features, index=self.idx)
        if self.target is not None:
            dataframe['target'] = self.target
        dataframe.to_csv(path_to_save)

    @property
    def memory_usage(self):
        if isinstance(self.features, np.ndarray):
            return sum([feature.nbytes for feature in self.features.T])
        else:
            return self.features.memory_usage().sum()


@dataclass
class InputData(Data):
    """Data class for input data for the nodes
    """

    def __post_init__(self):
        if self.numerical_idx is None:
            if self.features is not None and isinstance(self.features, np.ndarray) and self.features.ndim > 1:
                if self.categorical_idx is None:
                    self.numerical_idx = np.arange(0, self.features.shape[1])
                else:
                    self.numerical_idx = np.setdiff1d(np.arange(0, self.features.shape[1]), self.categorical_idx)
            else:
                self.numerical_idx = np.array([0])

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

    def subset_range(self, start: int, end: int):
        if not (0 <= start <= end <= len(self.idx)):
            raise ValueError('Incorrect boundaries for subset')
        new_features = new_target = None
        if self.features is not None:
            new_features = self.features[start:end + 1]
        if self.target is not None:
            new_target = self.target[start:end + 1]
        return InputData(idx=self.idx[start:end + 1], features=new_features,
                         target=new_target,
                         task=self.task, data_type=self.data_type)

    def subset_indices(self, selected_idx: List):
        """Get subset from :obj:`InputData` to extract all items with specified indices

        Args:
            selected_idx: ``list`` of indices for extraction

        Returns:
            :obj:`InputData`
        """

        idx_list = [str(i) for i in self.idx]

        # extractions of row number for each existing index from selected_idx
        row_nums = [idx_list.index(str(selected_ind)) for selected_ind in selected_idx
                    if str(selected_ind) in idx_list]
        new_features = new_target = None
        if self.features is not None:
            new_features = self.features[row_nums]
        if self.target is not None:
            new_target = self.target[row_nums]
        return InputData(idx=np.asarray(self.idx)[row_nums], features=new_features,
                         target=new_target,
                         task=self.task, data_type=self.data_type)

    def subset_features(self, feature_ids: np.array) -> Optional[InputData]:
        """
        Return new :obj:`InputData` with subset of features based on non-empty ``features_ids`` list or `None` otherwise
        """
        if feature_ids is None or feature_ids.size == 0:
            return None
        if isinstance(self.features, np.ndarray):
            subsample_features = self.features[:, feature_ids]
        else:
            subsample_features = self.features.iloc[:, feature_ids]

        subsample_input = InputData(
            features=subsample_features,
            data_type=self.data_type,
            target=self.target, task=self.task,
            idx=self.idx,
            categorical_idx=np.setdiff1d(self.categorical_idx, feature_ids),
            numerical_idx=np.setdiff1d(self.numerical_idx, feature_ids),
            encoded_idx=np.setdiff1d(self.encoded_idx, feature_ids),
            categorical_features=self.categorical_features,
            features_names=self.features_names,
            supplementary_data=self.supplementary_data
        )

        return subsample_input

    def shuffle(self):
        """Shuffles features and target if possible
        """

        if self.data_type in (DataTypesEnum.table, DataTypesEnum.image, DataTypesEnum.text):
            shuffled_ind = np.random.permutation(len(self.features))
            idx, features, target = np.asarray(self.idx)[shuffled_ind], self.features[shuffled_ind], self.target[
                shuffled_ind]
            self.idx = idx
            self.features = features
            self.target = target
        else:
            pass

    def convert_non_int_indexes_for_fit(self, pipeline):
        """Conversion non ``int`` (``datetime``, ``string``, etc) indexes in ``integer`` form on the fit stage
        """

        copied_data = deepcopy(self)
        is_timestamp = isinstance(copied_data.idx[0], pd._libs.tslibs.timestamps.Timestamp)
        is_numpy_datetime = isinstance(copied_data.idx[0], np.datetime64)
        # if fit stage-just creating range of integers
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

    def get_not_encoded_data(self):
        new_features, new_features_names = None, None
        new_num_idx, new_cat_idx = None, None
        num_features, cat_features = None, None
        num_features_names, cat_features_names = None, None

        # Checking numerical data exists
        if self.numerical_idx is not None and self.numerical_idx.size != 0:
            if isinstance(self.features, np.ndarray):
                num_features = self.features[:, self.numerical_idx]
            else:
                num_features = self.features.iloc[:, self.numerical_idx].to_numpy()

            if self.features_names is not None and np.size(self.features_names):
                num_features_names = self.features_names[self.numerical_idx]
            else:
                num_features_names = np.array([f'num_feature_{i}' for i in range(1, num_features.shape[1] + 1)])

        # Checking categorical data exists
        if self.categorical_idx is not None and self.categorical_idx.size != 0:
            cat_features = self.categorical_features

            if self.features_names is not None and np.size(self.features_names):
                cat_features_names = self.features_names[self.categorical_idx]
            else:
                cat_features_names = np.array([f'cat_feature_{i}' for i in range(1, cat_features.shape[1] + 1)])

        if num_features is not None and cat_features is not None:
            new_features = np.hstack((num_features, cat_features))
            new_features_names = np.hstack((num_features_names, cat_features_names))
            new_features_idx = np.array(range(new_features.shape[1]))
            new_num_idx = new_features_idx[:num_features.shape[1]]
            new_cat_idx = new_features_idx[-cat_features.shape[1]:]

        elif cat_features is not None:
            new_features = cat_features
            new_features_names = cat_features_names
            new_cat_idx = np.array(range(new_features.shape[1]))

        elif num_features is not None:
            new_features = num_features
            new_features_names = num_features_names
            new_num_idx = np.array(range(new_features.shape[1]))
        else:
            raise ValueError('There is no features')

        if isinstance(new_features, pd.DataFrame):
            new_features.columns = new_features_names

        return InputData(idx=self.idx, features=new_features, features_names=new_features_names,
                         numerical_idx=new_num_idx, categorical_idx=new_cat_idx,
                         target=self.target, task=self.task, data_type=self.data_type)

    @staticmethod
    def _resolve_func(pipeline, x):
        return pipeline.last_idx_int + (x - pipeline.last_idx_dt) // pipeline.period

    def _resolve_non_int_idx(self, pipeline):
        return np.array(list(map(lambda x: self._resolve_func(pipeline, x), self.idx)))


@dataclass
class OutputData(Data):
    """``Data`` type for data prediction in the node
    """

    features: Optional[Union[np.ndarray, pd.DataFrame]] = None
    predict: Optional[np.ndarray] = None
    target: Optional[np.ndarray] = None
    encoded_idx: Optional[np.ndarray] = None

    def save_predict(self, path_to_save: PathType) -> PathType:
        prediction = self.predict.tolist() if len(self.predict.shape) >= 2 else self.predict
        prediction_df = pd.DataFrame({'Index': self.idx, 'Prediction': prediction})
        try:
            prediction_df.to_csv(path_to_save, index=False)
        except (FileNotFoundError, PermissionError, OSError):
            path_to_save = './predictions.csv'
            prediction_df.to_csv(path_to_save, index=False)

        return Path(path_to_save).resolve()


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


def data_type_is_table(data: Union[InputData, OutputData]) -> bool:
    return data.data_type is DataTypesEnum.table


def data_type_is_ts(data: InputData) -> bool:
    return data.data_type is DataTypesEnum.ts


def data_type_is_multi_ts(data: InputData) -> bool:
    return data.data_type is DataTypesEnum.multi_ts


def data_type_is_text(data: InputData) -> bool:
    return data.data_type is DataTypesEnum.text


def data_type_is_image(data: InputData) -> bool:
    return data.data_type is DataTypesEnum.image


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
                        target_array: Optional[np.ndarray] = None,
                        idx: Optional[np.ndarray] = None,
                        task: Task = Task(TaskTypesEnum.classification),
                        data_type: Optional[DataTypesEnum] = None,
                        features_names: np.ndarray[str] = None,
                        categorical_idx: Union[list[int, str], np.ndarray[int, str]] = None) -> InputData:
    if idx is None:
        idx = np.arange(len(features_array))
    if data_type is None:
        data_type = autodetect_data_type(task)

    categorical_features = None
    if categorical_idx is not None:
        if isinstance(categorical_idx, list):
            categorical_idx = np.array(categorical_idx)

        if categorical_idx.size != 0 and isinstance(categorical_idx[0], str) and features_names is None:
            raise ValueError(
                'Impossible to specify categorical features by name when the features_names are not specified'
            )

        if categorical_idx.size != 0 and isinstance(categorical_idx[0], str):
            categorical_idx = np.array(
                [idx for idx, column in enumerate(features_names) if column in set(categorical_idx)]
            )

        if categorical_idx.size != 0:
            categorical_features = features_array[:, categorical_idx]

    data = InputData(
        idx=idx,
        features=features_array,
        target=target_array,
        features_names=features_names,
        categorical_idx=categorical_idx,
        categorical_features=categorical_features,
        task=task,
        data_type=data_type
    )

    return data


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
