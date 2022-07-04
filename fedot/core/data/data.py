import glob
import os
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fedot.utilities.requirements_notificator import warn_requirement

try:
    import cv2
except ModuleNotFoundError:
    warn_requirement('opencv-python')
    cv2 = None

from fedot.core.data.array_utilities import atleast_2d
from fedot.core.data.load_data import JSONBatchLoader, TextBatchLoader
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@dataclass
class Data:
    """
    Base Data type class
    """

    idx: np.array
    task: Task
    data_type: DataTypesEnum
    features: np.array
    target: Optional[np.array] = None

    # Object with supplementary info
    supplementary_data: SupplementaryData = field(default_factory=SupplementaryData)

    @staticmethod
    def from_csv(file_path=None,
                 delimiter=',',
                 task: Task = Task(TaskTypesEnum.classification),
                 data_type: DataTypesEnum = DataTypesEnum.table,
                 columns_to_drop: Optional[List] = None,
                 target_columns: Union[str, List] = '',
                 index_col: Optional[Union[str, int]] = 0):
        """Import data from ``csv``

        Args:
            file_path: the path to the ``CSV`` with data
            columns_to_drop: the names of columns that should be dropped
            delimiter: the delimiter to separate the columns
            task: the :obj:`Task` that should be solved with data
            data_type: the type of data interpretation
            target_columns: name of target column (last column if empty and no target if ``None``)
            index_col: column name or index to use as the :obj:`Data.idx`;\n
                if ``None`` then arrange new unique index

        Returns:
            data
        """

        data_frame = pd.read_csv(file_path, sep=delimiter, index_col=index_col)
        if columns_to_drop:
            data_frame = data_frame.drop(columns_to_drop, axis=1)

        idx = data_frame.index.to_numpy()
        features, target = process_target_and_features(data_frame, target_columns)

        return InputData(idx=idx, features=features, target=target, task=task, data_type=data_type)

    @staticmethod
    def from_csv_time_series(task: Task,
                             file_path=None,
                             delimiter=',',
                             is_predict=False,
                             target_column: Optional[str] = ''):
        df = pd.read_csv(file_path, sep=delimiter)

        idx = get_indices_from_file(df, file_path)

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

    @staticmethod
    def from_csv_multi_time_series(task: Task,
                                   file_path=None,
                                   delimiter=',',
                                   is_predict=False,
                                   columns_to_use: Optional[list] = None,
                                   target_column: Optional[str] = ''):
        """
        Forms :obj:`InputData` of ``multi_ts`` type from columns of different variant of the same variable

        Args:
            task: the :obj:`Task` that should be solved with data
            file_path: path to csv file
            delimiter: delimiter for pandas df
            is_predict: is preparing for fit or predict stage
            columns_to_use: ``list`` with names of columns of different variant of the same variable
            target_column: ``string`` with name of target column, used for predict stage

        Returns:
            data
        """

        df = pd.read_csv(file_path, sep=delimiter)

        idx = get_indices_from_file(df, file_path)
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
                   target_size: Optional[Tuple[int, int]] = None):
        """Input data from Image

        Args:
            images: the path to the directory with image data in ``np.ndarray`` format
                or array in ``np.ndarray`` format
            labels: the path to the directory with image labels in ``np.ndarray`` format
                or array in ``np.ndarray`` format
            task: the :obj:`Task` that should be solved with data
            target_size: size for the images resizing (if necessary)

        Returns:
            data
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
                            data_type: DataTypesEnum = DataTypesEnum.text):

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
                        data_type: DataTypesEnum = DataTypesEnum.text):

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
                        export_to_meta=False, is_multilabel=False, shuffle=True) -> 'InputData':
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
            combined dataset
        """

        if os.path.isfile(files_path):
            raise ValueError("""Path to the directory expected but got file""")

        df_data = JSONBatchLoader(path=files_path, label=label, fields_to_use=fields_to_use,
                                  shuffle=shuffle).extract(export_to_meta)

        if len(fields_to_use) > 1:
            fields_to_combine = []
            for field in fields_to_use:
                fields_to_combine.append(np.array(df_data[field]))
                # Unite if the element of text data is divided into strings
                if isinstance(df_data[field][0], list):
                    df_data[field] = [' '.join(piece) for piece in df_data[field]]

            features = np.column_stack(tuple(fields_to_combine))
        else:
            field = df_data[fields_to_use[0]]
            # process field with nested list
            if isinstance(field[0], list):
                field = [' '.join(piece) for piece in field]
            features = np.array(field)

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
        new_features = None
        if self.features is not None:
            new_features = self.features[start:end + 1]
        return InputData(idx=self.idx[start:end + 1], features=new_features,
                         target=self.target[start:end + 1],
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
        new_features = None

        if self.features is not None:
            new_features = self.features[row_nums]
        return InputData(idx=np.asarray(self.idx)[row_nums], features=new_features,
                         target=self.target[row_nums],
                         task=self.task, data_type=self.data_type)

    def subset_features(self, features_ids: list):
        """Return new :obj:`InputData` with subset of features based on ``features_ids`` list
        """

        subsample_features = self.features[:, features_ids]
        subsample_input = InputData(features=subsample_features,
                                    data_type=self.data_type,
                                    target=self.target, task=self.task,
                                    idx=self.idx,
                                    supplementary_data=self.supplementary_data)

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


def get_indices_from_file(data_frame, file_path, idx_column='datetime'):
    if idx_column in data_frame.columns:
        df = pd.read_csv(file_path,
                         parse_dates=[idx_column])
        idx = [str(d) for d in df[idx_column]]
        return idx
    return np.arange(0, len(data_frame))


def array_to_input_data(features_array: np.array,
                        target_array: np.array,
                        idx: Optional[np.array] = None,
                        task: Task = Task(TaskTypesEnum.classification),
                        data_type: Optional[DataTypesEnum] = None):
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
