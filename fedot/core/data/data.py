import os
import re
import warnings
from copy import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from fedot.core.algorithms.time_series.lagged_features import prepare_lagged_ts_for_prediction
from fedot.core.data.preprocessing import ImputationStrategy
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

        features = ImputationStrategy().fit(features).apply(features)

        return InputData(idx=idx, features=features, target=target, task=task, data_type=data_type)

    @staticmethod
    def from_text(files_path: str = None,
                  meta_file: str = None, lang: str = 'english',
                  label: str = 'label',
                  task: Task = Task(TaskTypesEnum.classification)):

        if files_path and meta_file:
            raise ValueError("""One of the params needed [files_path, meta_file] \
                             but got both""")
        if meta_file:
            features, target = TextData(path=meta_file,
                                        lang=lang, target=label).from_meta_file()
            idx = [index for index in range(len(target))]
            return InputData(idx=idx, features=features,
                             target=target, task=task, data_type=DataTypesEnum.table)
        elif files_path:
            pass
        else:
            raise ValueError("""One of the params needed [files_path, meta_file] \
                             but got none""")


class TextData:
    def __init__(self, path, lang: str, target: str = 'label'):
        self.path = path
        self.lang = lang.lower()
        self.stemmer = PorterStemmer
        self.lemmanizer = WordNetLemmatizer
        self.target = target

    def from_meta_file(self):
        if os.path.isdir(self.path):
            raise ValueError('Expected file but got direcotry')

        frame = pd.read_csv(self.path)
        frame = frame.sample(frac=1).reset_index(drop=True)
        messages = frame['text'].astype('U').tolist()

        clean_messages = []
        for message in messages:
            clean_message = self._words_preprocess(message)
            clean_messages.append(clean_message)
        messages.clear()
        tfidf_vectorizer = TfidfVectorizer()
        features = tfidf_vectorizer.fit_transform(clean_messages)
        target = frame[self.target]

        return features.toarray(), target

    def from_files(self):
        if os.path.isfile(self.path):
            raise ValueError('Expected directory with train and test directories but got file')

    def _words_preprocess(self, text):
        words = set(self._word_vectorize(text))
        without_stop_words = self._remove_stop_words(words)
        words = self._lemmatization(without_stop_words)
        words = [word for word in words if word.isalpha()]
        new_text = " ".join(words)

        return new_text

    def _clean_html_text(self, raw_text):
        clean_pattern = re.compile("<.*?>")
        text = re.sub(clean_pattern, " ", raw_text)

        return text

    def _word_vectorize(self, text):
        words = nltk.word_tokenize(text)

        return words

    def _remove_stop_words(self, words: set):
        stop_words = set(stopwords.words(self.lang))
        cleared_words = [word for word in words if word not in stop_words]

        return cleared_words

    def _stemming(self, words):
        stemmed_words = [self.stemmer().stem(word) for word in words]

        return stemmed_words

    def _lemmatization(self, words):
        # TODO pos
        lemmas = [self.lemmanizer().lemmatize(word) for word in words]

        return lemmas


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
    def from_predictions(outputs: List['OutputData'], target: np.array):
        if len(set([output.task.task_type for output in outputs])) > 1:
            raise ValueError('Inconsistent task types')

        task = outputs[0].task
        data_type = outputs[0].data_type
        idx = outputs[0].idx

        dataset_merging_funcs = {
            DataTypesEnum.forecasted_ts: _combine_datasets_ts,
            DataTypesEnum.ts: _combine_datasets_ts,
            DataTypesEnum.table: _combine_datasets_table
        }
        dataset_merging_funcs.setdefault(data_type, _combine_datasets_common)

        features = dataset_merging_funcs[data_type](outputs)

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

    def prepare_for_modelling(self, is_for_fit: bool = False):
        prepared_data = self
        if (self.data_type == DataTypesEnum.ts_lagged_table or
                self.data_type == DataTypesEnum.forecasted_ts):
            prepared_data = prepare_lagged_ts_for_prediction(self, is_for_fit)
        elif self.data_type in [DataTypesEnum.table, DataTypesEnum.forecasted_ts]:
            # TODO implement NaN filling here
            pass

        return prepared_data


@dataclass
class OutputData(Data):
    """
    Data type for data predicted in the node
    """
    predict: np.array = None


def split_train_test(data, split_ratio=0.8, with_shuffle=False, task: Task = None):
    assert 0. <= split_ratio <= 1.
    if task is not None and task.task_type == TaskTypesEnum.ts_forecasting:
        split_point = int(len(data) * split_ratio)
        # move pre-history of time series from train to test sample
        data_train, data_test = (data[:split_point],
                                 copy(data[split_point - task.task_params.max_window_size:]))
    else:
        if with_shuffle:
            data_train, data_test = train_test_split(data, test_size=1. - split_ratio, random_state=42)
        else:
            split_point = int(len(data) * split_ratio)
            data_train, data_test = data[:split_point], data[split_point:]
    return data_train, data_test


def _convert_dtypes(data_frame: pd.DataFrame):
    objects: pd.DataFrame = data_frame.select_dtypes('object')
    for column_name in objects:
        warnings.warn(f'Automatic factorization for the column {column_name} with type "object" is applied.')
        encoded = pd.factorize(data_frame[column_name])[0]
        data_frame[column_name] = encoded
    data_frame = data_frame.fillna(0)
    return data_frame


def train_test_data_setup(data: InputData, split_ratio=0.8,
                          shuffle_flag=False, task: Task = None) -> Tuple[InputData, InputData]:
    if data.features is not None:
        train_data_x, test_data_x = split_train_test(data.features, split_ratio, with_shuffle=shuffle_flag, task=task)
    else:
        train_data_x, test_data_x = None, None

    train_data_y, test_data_y = split_train_test(data.target, split_ratio, with_shuffle=shuffle_flag, task=task)
    train_idx, test_idx = split_train_test(data.idx, split_ratio, with_shuffle=shuffle_flag, task=task)
    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=train_idx, task=data.task, data_type=data.data_type)
    test_data = InputData(features=test_data_x, target=test_data_y, idx=test_idx, task=data.task,
                          data_type=data.data_type)
    return train_data, test_data


def _combine_datasets_ts(outputs: List[OutputData]):
    features_list = list()

    expected_len = max([len(output.predict) for output in outputs])

    for elem in outputs:
        predict = elem.predict
        if len(elem.predict) != expected_len:
            raise ValueError(f'Non-equal prediction length: {len(elem.predict)} and {expected_len}')
        features_list.append(predict)

    if len(features_list) > 1:
        features = np.column_stack(features_list)
    else:
        features = features_list[0]

    return features


def _combine_datasets_table(outputs: List[OutputData]):
    features = list()
    expected_len = len(outputs[0].predict)

    for elem in outputs:
        if len(elem.predict) != expected_len:
            raise ValueError(f'Non-equal prediction length: {len(elem.predict)} and {expected_len}')
        if len(elem.predict.shape) == 1:
            features.append(elem.predict)
        else:
            # if the model prediction is multivariate
            number_of_variables_in_prediction = elem.predict.shape[1]
            for i in range(number_of_variables_in_prediction):
                features.append(elem.predict[:, i])

    features = np.array(features).T

    return features


def _combine_datasets_common(outputs: List[OutputData]):
    features = list()

    for elem in outputs:
        if len(elem.predict) != len(outputs[0].predict):
            raise NotImplementedError(f'Non-equal prediction length: '
                                      f'{len(elem.predict)} and {len(outputs[0].predict)}')
        if len(elem.predict.shape) == 1:
            features.append(elem.predict)
        else:
            # if the model prediction is multivariate
            number_of_variables_in_prediction = elem.predict.shape[1]
            for i in range(number_of_variables_in_prediction):
                features.append(elem.predict[:, i])
    return features
