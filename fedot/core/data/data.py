import glob
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import imageio
import numpy as np
import pandas as pd
from PIL import Image

from fedot.core.data.load_data import JSONBatchLoader, TextBatchLoader
from fedot.core.data.merge import DataMerger
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.supplementary_data import SupplementaryData


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

        if target_column and target_column in data_frame.columns:
            target = np.array(data_frame[target_column]).astype(np.float)
            pos = list(data_frame.keys()).index(target_column)
            features = np.delete(data_array.T, [0, pos], axis=1)
        else:
            # no target in data
            features = data_array[1:].T
            target = None

        target = np.array(target)
        if len(target.shape) < 2:
            target = target.reshape((-1, 1))
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
            # Prepare InputData for train the chain
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


@dataclass
class OutputData(Data):
    """
    Data type for data prediction in the node
    """
    predict: np.array = None
    target: Optional[np.array] = None


def _convert_dtypes(data_frame: pd.DataFrame):
    """ Function converts columns with objects into numerical form and fill na """
    objects: pd.DataFrame = data_frame.select_dtypes('object')
    for column_name in objects:
        warnings.warn(f'Automatic factorization for the column {column_name} with type "object" is applied.')
        encoded = pd.factorize(data_frame[column_name])[0]
        data_frame[column_name] = encoded
    data_frame = data_frame.fillna(0)
    return data_frame


def _resize_image(file_path: str, target_size: tuple):
    im = Image.open(file_path)
    im_resized = im.resize(target_size, Image.NEAREST)
    im_resized.save(file_path, 'jpeg')

    img = np.asarray(imageio.imread(file_path, 'jpeg'))
    if len(img.shape) == 3:
        # TODO refactor for multi-color
        img = img[..., 0] + img[..., 1] + img[..., 2]
    return img
