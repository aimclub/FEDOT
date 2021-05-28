from typing import Tuple, Iterator

import numpy as np
from sklearn.model_selection import train_test_split, KFold

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum


def _split_time_series(data, task, *args, **kwargs):
    """ Split time series data into train and test parts

    :param data: array with data to split (not InputData)
    :param task: task to solve
    """

    input_features = data.features
    input_target = data.target
    forecast_length = task.task_params.forecast_length

    # Source time series divide into two parts
    x_train = input_features[:-forecast_length]
    x_test = input_features[:-forecast_length]

    y_train = x_train
    y_test = input_target[-forecast_length:]

    idx_for_train = np.arange(0, len(x_train))

    # Define indices for test
    start_forecast = len(x_train)
    end_forecast = start_forecast + forecast_length
    idx_for_predict = np.arange(start_forecast, end_forecast)

    # Prepare data to train the operation
    train_data = InputData(idx=idx_for_train,
                           features=x_train,
                           target=y_train,
                           task=task,
                           data_type=DataTypesEnum.ts)

    test_data = InputData(idx=idx_for_predict,
                          features=x_test,
                          target=y_test,
                          task=task,
                          data_type=DataTypesEnum.ts)

    return train_data, test_data


def _split_any(data, task, data_type, split_ratio, with_shuffle=False):
    """ Split any data into train and test parts

    :param data: array with data to split (not InputData)
    :param task: task to solve
    :param split_ratio: threshold for partitioning
    :param data_type type of data to split
    :param with_shuffle: is data needed to be shuffled or not
    """

    if not 0. < split_ratio < 1.:
        raise ValueError('Split ratio must belong to the interval (0; 1)')
    random_state = 42

    # Predictors and target
    input_features = data.features
    input_target = data.target

    x_train, x_test, y_train, y_test = train_test_split(input_features,
                                                        input_target,
                                                        test_size=1. - split_ratio,
                                                        shuffle=with_shuffle,
                                                        random_state=random_state)

    idx_for_train = np.arange(0, len(x_train))
    idx_for_predict = np.arange(0, len(x_test))

    # Prepare data to train the operation
    train_data = InputData(idx=idx_for_train,
                           features=x_train,
                           target=y_train,
                           task=task,
                           data_type=data_type)

    test_data = InputData(idx=idx_for_predict,
                          features=x_test,
                          target=y_test,
                          task=task,
                          data_type=data_type)

    return train_data, test_data


def _split_table(data, task, split_ratio, with_shuffle=False):
    """ Split table data into train and test parts

    :param data: array with data to split (not InputData)
    :param task: task to solve
    :param split_ratio: threshold for partitioning
    :param with_shuffle: is data needed to be shuffled or not
    """
    return _split_any(data, task, DataTypesEnum.table, split_ratio, with_shuffle)


def _split_image(data, task, split_ratio, with_shuffle=False):
    """ Split image data into train and test parts

    :param data: array with data to split (not InputData)
    :param task: task to solve
    :param split_ratio: threshold for partitioning
    :param with_shuffle: is data needed to be shuffled or not
    """

    return _split_any(data, task, DataTypesEnum.image, split_ratio, with_shuffle)


def _split_text(data, task, split_ratio, with_shuffle=False):
    """ Split text data into train and test parts

    :param data: array with data to split (not InputData)
    :param task: task to solve
    :param split_ratio: threshold for partitioning
    :param with_shuffle: is data needed to be shuffled or not
    """

    return _split_any(data, task, DataTypesEnum.image, split_ratio, with_shuffle)


def train_test_data_setup(data: InputData, split_ratio=0.8,
                          shuffle_flag=False) -> Tuple[InputData, InputData]:
    """ Function for train and test split

    :param data: InputData for train and test splitting
    :param split_ratio: threshold for partitioning
    :param shuffle_flag: is data needed to be shuffled or not

    :return train_data: InputData for train
    :return test_data: InputData for validation
    """
    # Split into train and test
    if data.features is not None:
        task = data.task

        split_func_dict = {
            DataTypesEnum.ts: _split_time_series,
            DataTypesEnum.table: _split_table,
            DataTypesEnum.image: _split_image,
            DataTypesEnum.text: _split_text
        }

        split_func = split_func_dict.get(data.data_type, _split_table)

        train_data, test_data = split_func(data, task, split_ratio,
                                           with_shuffle=shuffle_flag)
    else:
        raise ValueError('InputData must be not empty')

    return train_data, test_data


def train_test_multi_modal_data_setup(data: MultiModalData, split_ratio=0.8,
                                      shuffle_flag=False) -> Tuple[MultiModalData, MultiModalData]:
    train_data = MultiModalData()
    test_data = MultiModalData()
    for node in data.keys():
        data_part = data[node]
        train_data_part, test_data_part = train_test_data_setup(data_part, split_ratio, shuffle_flag)
        train_data[node] = train_data_part
        test_data[node] = test_data_part

    return train_data, test_data


def train_test_cv_generator(data: InputData, folds: int) -> Iterator[Tuple[InputData, InputData]]:
    """ The function for splitting data into a train and test samples
        in the InputData format for KFolds cross validation. The function
        return a generator of tuples, consisting of a pair of train, test.

    :param data: InputData for train and test splitting
    :param folds: number of folds

    :return Iterator[InputData, InputData]: return split train/test data
    """
    kf = KFold(n_splits=folds)

    for train_idxs, test_idxs in kf.split(data.features):
        train_features, train_target = \
            _features_and_target_by_index(train_idxs, data)
        test_features, test_target = \
            _features_and_target_by_index(test_idxs, data)

        idx_for_train = np.arange(0, len(train_features))
        idx_for_test = np.arange(0, len(test_features))

        train_data = InputData(idx=idx_for_train,
                               features=train_features,
                               target=train_target,
                               task=data.task,
                               data_type=data.data_type)
        test_data = InputData(idx=idx_for_test,
                              features=test_features,
                              target=test_target,
                              task=data.task,
                              data_type=data.data_type)

        yield train_data, test_data


def _features_and_target_by_index(index, values: InputData):
    features = values.features[index, :]
    target = np.take(values.target, index)

    return features, target
