import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import imageio
import numpy as np
import pandas as pd
from PIL import Image

from fedot.core.data.load_data import JSONBatchLoader, TextBatchLoader
from fedot.core.data.merge import DataMerger
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

# Max unique values to convert numerical column to categorical.
MAX_UNIQ_VAL = 12


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
    supplementary_data: SupplementaryData = SupplementaryData()

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
            # Prepare InputData for train the pipeline
            input_data = InputData(idx=np.arange(0, len(time_series)),
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
                        export_to_meta=False) -> 'InputData':
        """
        Generates InputData from the set of JSON files with different fields
        :param files_path: path the folder with jsons
        :param fields_to_use: list of fields that will be considered as a features
        :param label: name of field with target variable
        :param task: task to solve
        :param data_type: data type in fields (as well as type for obtained InputData)
        :param export_to_meta: combine extracted field and save to CSV
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
                val = [v[0] for v in val]
            features = np.array(val)

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
        if self.task.task_type == TaskTypesEnum.classification:
            return len(np.unique(self.target))
        else:
            return None

    @staticmethod
    def from_predictions(outputs: List['OutputData']):
        """ Method obtain predictions from previous nodes """
        # Update not only features but idx, target and task also
        idx, features, target, task, d_type, updated_info = DataMerger(outputs).merge()

        return InputData(idx=idx, features=features, target=target, task=task,
                         data_type=d_type, supplementary_data=updated_info)

    def subset(self, start: int, end: int):
        if not (0 <= start <= end <= len(self.idx)):
            raise ValueError('Incorrect boundaries for subset')
        new_features = None
        if self.features is not None:
            new_features = self.features[start:end + 1]
        return InputData(idx=self.idx[start:end + 1], features=new_features,
                         target=self.target[start:end + 1], task=self.task, data_type=self.data_type)

    def shuffle(self):
        """
        Shuffles features and target if possible
        """
        if self.data_type == DataTypesEnum.table:
            shuffled_ind = np.random.permutation(len(self.features))
            idx, features, target = self.idx[shuffled_ind], self.features[shuffled_ind], self.target[shuffled_ind]
            self.idx = idx
            self.features = features
            self.target = target
        else:
            pass


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


def data_has_categorical_features(data: Union[InputData, MultiModalData]) -> bool:
    """ Check data for categorical columns. Also check, if some numerical column
    has unique values less then MAX_UNIQ_VAL, then convert this column to string.

    :param data: Union[InputData, MultiModalData]
    :return data_has_categorical_columns: bool, whether data has categorical columns or not
    """

    data_has_categorical_columns = False

    if isinstance(data, MultiModalData):
        for data_source_name, values in data.items():
            if data_source_name.startswith('data_source_table'):
                data_has_categorical_columns = _has_data_categorical(values)
    elif data_type_is_suitable_preprocessing(data):
        data_has_categorical_columns = _has_data_categorical(data)

    return data_has_categorical_columns


def data_has_missing_values(data: Union[InputData, MultiModalData]) -> bool:
    """ Check data for missing values."""

    if isinstance(data, MultiModalData):
        for data_source_name, values in data.items():
            if data_type_is_table(values):
                return pd.DataFrame(values.features).isna().sum().sum() > 0
    elif data_type_is_suitable_preprocessing(data):
        return pd.DataFrame(data.features).isna().sum().sum() > 0
    return False


def str_columns_check(features):
    """
    Method for checking which columns contain categorical (text) data

    :param features: tabular data for check
    :return categorical_ids: indices of categorical columns in table
    :return non_categorical_ids: indices of non categorical columns in table
    """
    source_shape = features.shape
    columns_amount = source_shape[1] if len(source_shape) > 1 else 1

    categorical_ids = []
    non_categorical_ids = []
    # For every column in table make check for first element
    for column_id in range(0, columns_amount):
        column = features[:, column_id] if columns_amount > 1 else features
        if isinstance(column[0], str):
            categorical_ids.append(column_id)
        else:
            non_categorical_ids.append(column_id)

    return categorical_ids, non_categorical_ids


def divide_data_categorical_numerical(input_data: InputData) -> (InputData, InputData):
    categorical_ids, non_categorical_ids = str_columns_check(input_data.features)
    numerical_features = input_data.features[:, non_categorical_ids]
    categorical_features = input_data.features[:, categorical_ids]

    numerical = InputData(features=numerical_features, data_type=input_data.data_type,
                          target=input_data.target, task=input_data.task, idx=input_data.idx)
    categorical = InputData(features=categorical_features, data_type=input_data.data_type,
                            target=input_data.target, task=input_data.task, idx=input_data.idx)

    return numerical, categorical


def data_type_is_table(data: InputData) -> bool:
    return data.data_type == DataTypesEnum.table


def data_type_is_suitable_preprocessing(data: InputData) -> bool:
    if data.data_type == DataTypesEnum.table or data.data_type == DataTypesEnum.ts:
        return True
    return False


def _has_data_categorical(data: InputData) -> bool:
    """ Whether data categorical columns or not.

    :param data: InputData
    :return data_has_categorical_columns: bool, whether data has categorical columns or not
    """
    data_has_categorical_columns = False

    if isinstance(data.features, list) or len(data.features.shape) == 1:
        data_has_categorical_columns = _is_values_categorical(data.features)
    else:
        num_columns = data.features.shape[1]
        for col_index in range(num_columns):
            if data_has_categorical_columns:
                break
            data_has_categorical_columns = _is_values_categorical(data.features[:, col_index])

    return data_has_categorical_columns


def _is_values_categorical(values: List):
    # Check if any value in list has 'string' type
    return any(list(map(lambda x: isinstance(x, str), values)))
