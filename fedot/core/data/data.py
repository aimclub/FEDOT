import glob
import os
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import imageio
import numpy as np
import pandas as pd
from PIL import Image

from fedot.core.data.load_data import JSONBatchLoader, TextBatchLoader
from fedot.core.data.merge import DataMerger
from fedot.core.data.supplementary_data import SupplementaryData
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
    # Object with supplementary info
    supplementary_data: SupplementaryData = field(default_factory=SupplementaryData)

    @staticmethod
    def from_csv(file_path=None,
                 delimiter=',',
                 task: Task = Task(TaskTypesEnum.classification),
                 data_type: DataTypesEnum = DataTypesEnum.table,
                 columns_to_drop: Optional[List] = None,
                 target_columns: Union[str, List] = ''):
        """
        :param file_path: the path to the CSV with data
        :param columns_to_drop: the names of columns that should be dropped
        :param delimiter: the delimiter to separate the columns
        :param task: the task that should be solved with data
        :param data_type: the type of data interpretation
        :param target_columns: name of target column (last column if empty and no target if None)
        :return:
        """

        data_frame = pd.read_csv(file_path, sep=delimiter)
        if columns_to_drop:
            data_frame = data_frame.drop(columns_to_drop, axis=1)

        # Get indices of the DataFrame
        data_array = np.array(data_frame).T
        idx = data_array[0]
        if isinstance(idx[0], float) and idx[0] == round(idx[0]):
            # if float indices is unnecessary
            idx = [str(round(i)) for i in idx]
        if type(target_columns) is list:
            features, target = process_multiple_columns(target_columns, data_frame)
        else:
            features, target = process_one_column(target_columns, data_frame,
                                                  data_array)

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
    def from_image(images: Union[str, np.ndarray] = None,
                   labels: Union[str, np.ndarray] = None,
                   task: Task = Task(TaskTypesEnum.classification),
                   target_size: Optional[Tuple[int, int]] = None):
        """
        :param images: the path to the directory with image data in np.ndarray format or array in np.ndarray format
        :param labels: the path to the directory with image labels in np.ndarray format or array in np.ndarray format
        :param task: the task that should be solved with data
        :param target_size: size for the images resizing (if necessary)
        :return:
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

    @staticmethod
    def from_json_files(files_path: str,
                        fields_to_use: List,
                        label: str = 'label',
                        task: Task = Task(TaskTypesEnum.classification),
                        data_type: DataTypesEnum = DataTypesEnum.table,
                        export_to_meta=False, is_multilabel=False) -> 'InputData':
        """
        Generates InputData from the set of JSON files with different fields
        :param files_path: path the folder with jsons
        :param fields_to_use: list of fields that will be considered as a features
        :param label: name of field with target variable
        :param task: task to solve
        :param data_type: data type in fields (as well as type for obtained InputData)
        :param export_to_meta: combine extracted field and save to CSV
        :param is_multilabel: if True, creates multilabel target
        :return: combined dataset
        """

        if os.path.isfile(files_path):
            raise ValueError("""Path to the directory expected but got file""")

        df_data = JSONBatchLoader(path=files_path, label=label, fields_to_use=fields_to_use).extract(export_to_meta)

        if len(fields_to_use) > 1:
            fields_to_combine = []
            for f in fields_to_use:
                fields_to_combine.append(np.array(df_data[f]))

            features = np.column_stack(tuple(fields_to_combine))
        else:
            val = df_data[fields_to_use[0]]
            # process field with nested list
            if isinstance(val[0], list):
                val = [' '.join(v) for v in val]
            features = np.array(val)

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


@dataclass
class InputData(Data):
    """
    Data class for input data for the nodes
    """
    target: Optional[np.array] = None

    @property
    def num_classes(self) -> Optional[int]:
        if self.task.task_type == TaskTypesEnum.classification and self.target is not None:
            return len(np.unique(self.target))
        else:
            return None

    @staticmethod
    def from_predictions(outputs: List['OutputData']):
        """ Method obtain predictions from previous nodes """
        # Update not only features but idx, target and task also
        idx, features, target, task, d_type, updated_info = DataMerger(outputs).merge()

        # Mark data as preprocessed already
        updated_info.was_preprocessed = True
        return InputData(idx=idx, features=features, target=target, task=task,
                         data_type=d_type, supplementary_data=updated_info)

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
        """
        Get subset from InputData to extract all items with specified indices
        :param selected_idx: list of indices for extraction
        :return:
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
        """ Return new InputData with subset of features based on features_ids list """
        subsample_features = self.features[:, features_ids]
        subsample_input = InputData(features=subsample_features,
                                    data_type=self.data_type,
                                    target=self.target, task=self.task,
                                    idx=self.idx,
                                    supplementary_data=self.supplementary_data)

        return subsample_input

    def shuffle(self):
        """
        Shuffles features and target if possible
        """
        if self.data_type == DataTypesEnum.table:
            shuffled_ind = np.random.permutation(len(self.features))
            idx, features, target = np.asarray(self.idx)[shuffled_ind], self.features[shuffled_ind], self.target[
                shuffled_ind]
            self.idx = idx
            self.features = features
            self.target = target
        else:
            pass

    def convert_non_int_indexes_for_fit(self, pipeline):
        """ Conversion non int (datetime, string, etc) indexes in integer form in fit stage """
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
        elif type(copied_data.idx[0]) not in [int, np.int32, np.int64]:
            copied_data.supplementary_data.non_int_idx = copy(copied_data.idx)
            copied_data.idx = np.array(range(len(copied_data.idx)))
            pipeline.last_idx_int = copied_data.idx[-1]
        return copied_data

    def convert_non_int_indexes_for_predict(self, pipeline):
        """Conversion non int (datetime, string, etc) indexes in integer form in predict stage"""
        copied_data = deepcopy(self)
        is_timestamp = isinstance(copied_data.idx[0], pd._libs.tslibs.timestamps.Timestamp)
        is_numpy_datetime = isinstance(copied_data.idx[0], np.datetime64)
        # if predict stage - calculating shift from last train part index
        if is_timestamp or is_numpy_datetime:
            copied_data.supplementary_data.non_int_idx = copy(self.idx)
            copied_data.idx = self._resolve_non_int_idx(pipeline)
        elif type(copied_data.idx[0]) not in [int, np.int32, np.int64]:
            # note, that string indexes do not have an order and always we think that indexes we want to predict go
            # immediately after the train indexes
            copied_data.supplementary_data.non_int_idx = copy(copied_data.idx)
            copied_data.idx = pipeline.last_idx_int + np.array(range(1, len(copied_data.idx)+1))
        return copied_data

    @staticmethod
    def _resolve_func(pipeline, x):
        return pipeline.last_idx_int + (x - pipeline.last_idx_dt) // pipeline.period

    def _resolve_non_int_idx(self, pipeline):
        return np.array(list(map(lambda x: self._resolve_func(pipeline, x), self.idx)))


@dataclass
class OutputData(Data):
    """
    Data type for data prediction in the node
    """
    predict: np.array = None
    target: Optional[np.array] = None


def _resize_image(file_path: str, target_size: tuple):
    im = Image.open(file_path)
    im_resized = im.resize(target_size, Image.NEAREST)
    im_resized.save(file_path, 'jpeg')

    img = np.asarray(imageio.imread(file_path, 'jpeg'))
    if len(img.shape) == 3:
        # TODO refactor for multi-color
        img = img[..., 0] + img[..., 1] + img[..., 2]
    return img


def process_one_column(target_column, data_frame, data_array):
    """ Function process pandas dataframe with single column

    :param target_column: name of column with target or None
    :param data_frame: loaded panda DataFrame
    :param data_array: array received from source DataFrame
    :return features: numpy array (table) with features
    :return target: numpy array (column) with target
    """
    if target_column == '':
        # Take the last column in the table
        target_column = data_frame.columns[-1]

    if target_column and target_column in data_frame.columns:
        target = np.array(data_frame[target_column])
        pos = list(data_frame.keys()).index(target_column)
        features = np.delete(data_array.T, [0, pos], axis=1)
    else:
        # no target in data
        features = data_array[1:].T
        target = None

    if target is not None:
        target = np.array(target)
        if len(target.shape) < 2:
            target = target.reshape((-1, 1))

    return features, target


def process_multiple_columns(target_columns, data_frame):
    """ Function for processing target """
    features = np.array(data_frame.drop(columns=target_columns))

    # Remove index column
    targets = np.array(data_frame[target_columns])

    return features, targets


def data_type_is_table(data: Union[InputData, OutputData]) -> bool:
    return data.data_type == DataTypesEnum.table


def data_type_is_ts(data: InputData) -> bool:
    return data.data_type == DataTypesEnum.ts


def get_indices_from_file(data_frame, file_path):
    if 'datetime' in data_frame.columns:
        df = pd.read_csv(file_path,
                         parse_dates=['datetime'])
        idx = [str(d) for d in df['datetime']]
        return idx
    return np.arange(0, len(data_frame))


def array_to_input_data(features_array: np.array,
                        target_array: np.array,
                        idx: Optional[np.array] = None,
                        task: Task = Task(TaskTypesEnum.classification)):
    data_type = autodetect_data_type(task)

    if idx is None:
        idx = np.arange(len(features_array))

    return InputData(idx=idx, features=features_array, target=target_array, task=task, data_type=data_type)


def autodetect_data_type(task: Task) -> DataTypesEnum:
    if task.task_type == TaskTypesEnum.ts_forecasting:
        return DataTypesEnum.ts
    else:
        return DataTypesEnum.table
