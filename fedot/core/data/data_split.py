from copy import deepcopy
from typing import Tuple, Union

from sklearn.model_selection import train_test_split

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum


def _split_time_series(data: InputData, task, *args, **kwargs):
    """ Split time series data into train and test parts

    :param data: InputData object to split
    :param task: task to solve
    """

    input_features = data.features
    input_target = data.target
    forecast_length = task.task_params.forecast_length

    if kwargs.get('validation_blocks') is not None:
        # It is required to split data for in-sample forecasting
        forecast_length = forecast_length * kwargs.get('validation_blocks')
        x_train = input_features[:-forecast_length]
        x_test = input_features

        y_train = input_target[:-forecast_length]
        y_test = input_target[-forecast_length:]
    else:
        # Source time series divide into two parts
        x_train = input_features[:-forecast_length]
        x_test = input_features[:-forecast_length]

        y_train = input_target[:-forecast_length]
        y_test = input_target[-forecast_length:]

    idx_train = data.idx[:-forecast_length]
    idx_test = data.idx[-forecast_length:]

    # Prepare data to train the operation
    train_data = InputData(idx=idx_train, features=x_train, target=y_train,
                           task=task, data_type=DataTypesEnum.ts,
                           supplementary_data=data.supplementary_data)

    test_data = InputData(idx=idx_test, features=x_test, target=y_test,
                          task=task, data_type=DataTypesEnum.ts,
                          supplementary_data=data.supplementary_data)
    return train_data, test_data


def _split_multi_time_series(data: InputData, task, *args, **kwargs):
    """ Split multi_ts time series data into train and test parts

    :param data: InputData object to split
    :param task: task to solve
    """

    input_features = data.features
    input_target = data.target
    forecast_length = task.task_params.forecast_length

    if kwargs.get('validation_blocks') is not None:
        # It is required to split data for in-sample forecasting
        forecast_length = forecast_length * kwargs.get('validation_blocks')
        x_train = input_features[:-forecast_length]
        x_test = input_features

        y_train = input_target[:-forecast_length]
        y_test = input_target[-forecast_length:, 0]

    else:
        # Source time series divide into two parts
        x_train = input_features[:-forecast_length]
        x_test = input_features[:-forecast_length]

        y_train = input_target[:-forecast_length]
        y_test = input_target[-forecast_length:, 0]

    idx_train = data.idx[:-forecast_length]
    idx_test = data.idx[-forecast_length:]

    # Prepare data to train the operation
    train_data = InputData(idx=idx_train, features=x_train, target=y_train,
                           task=task, data_type=DataTypesEnum.multi_ts,
                           supplementary_data=data.supplementary_data)

    test_data = InputData(idx=idx_test, features=x_test, target=y_test,
                          task=task, data_type=DataTypesEnum.multi_ts,
                          supplementary_data=data.supplementary_data)

    return train_data, test_data


def _split_any(data: InputData, task, data_type, split_ratio, with_shuffle=False, **kwargs):
    """ Split any data into train and test parts

    :param data: InputData object to split
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
    idx = data.idx
    idx_train, idx_test, x_train, x_test, y_train, y_test = \
        train_test_split(idx,
                         input_features,
                         input_target,
                         test_size=1. - split_ratio,
                         shuffle=with_shuffle,
                         random_state=random_state)

    # Prepare data to train the operation
    train_data = InputData(idx=idx_train, features=x_train,  target=y_train,
                           task=task, data_type=data_type,
                           supplementary_data=data.supplementary_data)

    test_data = InputData(idx=idx_test, features=x_test, target=y_test,
                          task=task, data_type=data_type,
                          supplementary_data=data.supplementary_data)

    return train_data, test_data


def _split_table(data: InputData, task, split_ratio, with_shuffle=False, **kwargs):
    """ Split table data into train and test parts

    :param data: InputData object to split
    :param task: task to solve
    :param split_ratio: threshold for partitioning
    :param with_shuffle: is data needed to be shuffled or not
    """
    return _split_any(data, task, DataTypesEnum.table, split_ratio, with_shuffle)


def _split_image(data: InputData, task, split_ratio, with_shuffle=False, **kwargs):
    """ Split image data into train and test parts

    :param data: InputData object to split
    :param task: task to solve
    :param split_ratio: threshold for partitioning
    :param with_shuffle: is data needed to be shuffled or not
    """

    return _split_any(data, task, DataTypesEnum.image, split_ratio, with_shuffle)


def _split_text(data: InputData, task, split_ratio, with_shuffle=False, **kwargs):
    """ Split text data into train and test parts

    :param data: InputData object to split
    :param task: task to solve
    :param split_ratio: threshold for partitioning
    :param with_shuffle: is data needed to be shuffled or not
    """

    return _split_any(data, task, DataTypesEnum.text, split_ratio, with_shuffle)


def _train_test_single_data_setup(data: InputData, split_ratio=0.8,
                                  shuffle_flag=False, **kwargs) -> Tuple[InputData, InputData]:
    """ Function for train and test split

    :param data: InputData for train and test splitting
    :param split_ratio: threshold for partitioning
    :param shuffle_flag: is data needed to be shuffled or not

    :return train_data: InputData for train
    :return test_data: InputData for validation
    """
    # Split into train and test
    if data is not None:
        task = data.task

        split_func_dict = {
            DataTypesEnum.multi_ts: _split_multi_time_series,
            DataTypesEnum.ts: _split_time_series,
            DataTypesEnum.table: _split_table,
            DataTypesEnum.image: _split_image,
            DataTypesEnum.text: _split_text
        }

        split_func = split_func_dict.get(data.data_type, _split_table)

        train_data, test_data = split_func(data, task, split_ratio,
                                           with_shuffle=shuffle_flag,
                                           **kwargs)
    else:
        raise ValueError('InputData must be not empty')

    # Store additional information
    train_data.supplementary_data = deepcopy(data.supplementary_data)
    test_data.supplementary_data = deepcopy(data.supplementary_data)
    return train_data, test_data


def _train_test_multi_modal_data_setup(data: MultiModalData, split_ratio=0.8,
                                       shuffle_flag=False, **kwargs) -> Tuple[MultiModalData, MultiModalData]:
    train_data = MultiModalData()
    test_data = MultiModalData()
    for node in data.keys():
        data_part = data[node]
        train_data_part, test_data_part = train_test_data_setup(data_part, split_ratio, shuffle_flag, **kwargs)
        train_data[node] = train_data_part
        test_data[node] = test_data_part

    return train_data, test_data


def train_test_data_setup(data: Union[InputData, MultiModalData], split_ratio=0.8,
                          shuffle_flag=False, **kwargs) -> Tuple[Union[InputData, MultiModalData],
                                                                 Union[InputData, MultiModalData]]:
    """ Function for train and test split for both InputData and MultiModalData

    Args:
        data: data for train and test splitting
        split_ratio: threshold for partitioning
        shuffle_flag: is data needed to be shuffled or not
        kwargs: additional optional parameters such as number of validation blocks

    Returns:
        data for train, data for validation
    """
    if isinstance(data, InputData):
        train_data, test_data = _train_test_single_data_setup(data, split_ratio, shuffle_flag, **kwargs)
    elif isinstance(data, MultiModalData):
        train_data, test_data = _train_test_multi_modal_data_setup(data, split_ratio, shuffle_flag, **kwargs)
    else:
        raise ValueError(f'Dataset {type(data)} is not supported')

    return train_data, test_data
