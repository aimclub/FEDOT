from copy import deepcopy
from typing import Tuple, Optional, Union

import numpy as np
from sklearn.model_selection import train_test_split

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum


def _split_input_data_by_indexes(origin_input_data: Union[InputData, MultiModalData],
                                 index,
                                 retain_first_target=False):
    """ The function get InputData or MultiModalData and return
        only data with indexes in index, not in idx
        f.e. index = [0, 1, 2, 3] == input_data.features[[0, 1, 2, 3], :]
        :param origin_input_data: data to split
        :param index: indexes that needed in output data
        :param retain_first_target: set to True for use only first column of target
        """

    if isinstance(origin_input_data, MultiModalData):
        data = MultiModalData()
        for key in origin_input_data:
            data[key] = _split_input_data_by_indexes(origin_input_data[key],
                                                     index=index,
                                                     retain_first_target=retain_first_target)
        return data
    elif isinstance(origin_input_data, InputData):
        idx = np.take(origin_input_data.idx, index, 0)
        target = np.take(origin_input_data.target, index, 0)
        features = np.take(origin_input_data.features, index, 0)
        categorical_features = np.take(origin_input_data.categorical_features, index, 0)

        if retain_first_target and len(target.shape) > 1:
            target = target[:, 0]

        data = InputData(idx=idx,
                         features=features,
                         target=target,
                         task=deepcopy(origin_input_data.task),
                         data_type=origin_input_data.data_type,
                         supplementary_data=origin_input_data.supplementary_data,
                         categorical_features=categorical_features,
                         categorical_idx=origin_input_data.categorical_idx,
                         numerical_idx=origin_input_data.numerical_idx,
                         encoded_idx=origin_input_data.encoded_idx,
                         features_names=origin_input_data.features_names,
                         )
        return data
    else:
        raise TypeError(f'Unknown data type {type(origin_input_data)}')


def _split_time_series(data: InputData,
                       validation_blocks: Optional[int] = None,
                       **kwargs):
    """ Split time series data into train and test parts

    :param data: InputData object to split
    :param validation_blocks: validation blocks are used for test
    """

    forecast_length = data.task.task_params.forecast_length
    if validation_blocks is not None:
        forecast_length *= validation_blocks

    target_length = len(data.target)
    train_data = _split_input_data_by_indexes(data, index=np.arange(0, target_length - forecast_length),)
    test_data = _split_input_data_by_indexes(data, index=np.arange(target_length - forecast_length, target_length),
                                             retain_first_target=True)

    if validation_blocks is None:
        # for in-sample
        test_data.features = train_data.features
    else:
        # for out-of-sample
        test_data.features = data.features

    return train_data, test_data


def _split_any(data: InputData,
               split_ratio: float,
               shuffle: bool,
               stratify: bool,
               random_seed: int,
               **kwargs):
    """ Split any data except timeseries into train and test parts

    :param data: InputData object to split
    :param split_ratio: share of train data between 0 and 1
    :param shuffle: is data needed to be shuffled or not
    :param stratify: make stratified sample or not
    :param random_seed: random_seed for shuffle
    """

    stratify_labels = data.target if stratify else None

    train_ids, test_ids = train_test_split(np.arange(0, len(data.target)),
                                           test_size=1. - split_ratio,
                                           shuffle=shuffle,
                                           random_state=random_seed,
                                           stratify=stratify_labels)

    train_data = _split_input_data_by_indexes(data, index=train_ids)
    test_data = _split_input_data_by_indexes(data, index=test_ids)

    return train_data, test_data


def _are_stratification_allowed(data: Union[InputData, MultiModalData], split_ratio: float) -> bool:
    """ Check that stratification may be done
        :param data: data for split
        :param split_ratio: relation between train data length and all data length
        :return bool: stratification is allowed"""

    # check task_type
    if data.task.task_type is not TaskTypesEnum.classification:
        return False

    try:
        # fast way
        classes = np.unique(data.target, return_counts=True)
    except Exception:
        # slow way
        from collections import Counter
        classes = Counter(data.target)
        classes = [list(classes), list(classes.values())]

    # check that there are enough labels for two samples
    if not all(x > 1 for x in classes[1]):
        if __debug__:
            # tests often use very small datasets that are not suitable for data splitting
            # stratification is disabled for tests
            return False
        else:
            raise ValueError(("There is the only value for some classes:"
                              f" {', '.join(str(val) for val, count in zip(*classes) if count == 1)}."
                              f" Data split can not be done for {data.task.task_type.name} task."))

    # check that split ratio allows to set all classes to both samples
    test_size = round(len(data.target) * (1. - split_ratio))
    labels_count = len(classes[0])
    if test_size < labels_count:
        return False

    return True


def train_test_data_setup(data: Union[InputData, MultiModalData],
                          split_ratio: float = 0.8,
                          shuffle: bool = False,
                          shuffle_flag: bool = False,
                          stratify: bool = True,
                          random_seed: int = 42,
                          validation_blocks: Optional[int] = None) -> Tuple[Union[InputData, MultiModalData],
                                                                            Union[InputData, MultiModalData]]:
    """ Function for train and test split for both InputData and MultiModalData

    :param data: InputData object to split
    :param split_ratio: share of train data between 0 and 1
    :param shuffle: is data needed to be shuffled or not
    :param shuffle_flag: same is shuffle, use for backward compatibility
    :param stratify: make stratified sample or not
    :param random_seed: random_seed for shuffle
    :param validation_blocks: validation blocks are used for test

    :return: data for train, data for validation
    """

    # for backward compatibility
    shuffle |= shuffle_flag
    # check that stratification may be done
    stratify &= _are_stratification_allowed(data, split_ratio)
    # stratification is allowed only with shuffle
    shuffle |= stratify
    # shuffle is allowed only with random_seed and vise versa
    random_seed = (random_seed or 42) if shuffle else None

    input_arguments = {'split_ratio': split_ratio,
                       'shuffle': shuffle,
                       'stratify': stratify,
                       'random_seed': random_seed,
                       'validation_blocks': validation_blocks}
    if isinstance(data, InputData):
        split_func_dict = {DataTypesEnum.multi_ts: _split_time_series,
                           DataTypesEnum.ts: _split_time_series,
                           DataTypesEnum.table: _split_any,
                           DataTypesEnum.image: _split_any,
                           DataTypesEnum.text: _split_any}

        if data.data_type not in split_func_dict:
            raise TypeError((f'Unknown data type {type(data)}. Supported data types:'
                             f' {", ".join(str(x) for x in split_func_dict)}'))

        split_func = split_func_dict[data.data_type]
        train_data, test_data = split_func(data, **input_arguments)
    elif isinstance(data, MultiModalData):
        train_data, test_data = MultiModalData(), MultiModalData()
        for node in data.keys():
            train_data[node], test_data[node] = train_test_data_setup(data[node], **input_arguments)
    else:
        raise ValueError((f'Dataset {type(data)} is not supported. Supported types:'
                          ' InputData, MultiModalData'))

    return train_data, test_data
