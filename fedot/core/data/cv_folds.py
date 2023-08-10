from typing import Iterator, Optional, Tuple, Union

import numpy as np

from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.tasks import TaskTypesEnum
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.model_selection._split import StratifiedKFold

from fedot.core.data.data import InputData
from fedot.core.data.data_split import _split_input_data_by_indexes


class TsInputDataSplit(TimeSeriesSplit):
    """ Perform time series splitting for cross validation on InputData structures.
    The difference between TimeSeriesSplit (sklearn) and TsInputDataSplit can be
    demonstrated by an example:
    The time series [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] must be splitted into 3
    parts, where the size of each fold for validation will be 2 elements.
    TimeSeriesSplit (return indices)
        train - [0, 1, 2, 3] test - [4, 5]
        train - [0, 1, 2, 3, 4, 5] test - [6, 7]
        train - [0, 1, 2, 3, 4, 5, 6, 7] test - [8, 9]
    TsInputDataSplit (return values of time series)
        train - [1, 2, 3, 4] test - [1, 2, 3, 4, 5, 6]
        train - [1, 2, 3, 4, 5, 6] test - [1, 2, 3, 4, 5, 6, 7, 8]
        train - [1, 2, 3, 4, 5, 6, 7, 8] test - [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """

    def __init__(self, n_splits: int, test_size: int):
        super().__init__(gap=0, n_splits=n_splits, test_size=test_size)

    def split(self, data: np.ndarray, *args) -> Iterator[Tuple[InputData, InputData]]:
        """ Define indexes for train and validation using
        "in-sample forecasting" algorithm

        :param data: InputData for splitting
        """

        for train_ids, test_ids in super().split(data):
            new_test_ids = np.hstack((train_ids, test_ids))
            yield train_ids, new_test_ids


def cv_generator(data: Union[InputData, MultiModalData],
                 cv_folds: int,
                 shuffle: bool = False,
                 random_seed: int = 42,
                 stratify: bool = True,
                 validation_blocks: Optional[int] = None) -> Iterator[Tuple[Union[InputData, MultiModalData],
                                                                            Union[InputData, MultiModalData]]]:
    """ The function for splitting data into a train and test samples
        in the InputData format for cross validation. The function
        return a generator of tuples, consisting of a pair of train, test.

    :param data: InputData for train and test splitting
    :param shuffle: is data need shuffle
    :param cv_folds: number of folds
    :param random_seed: random seed for shuffle
    :param stratify: `True` to make stratified samples for classification task
    :param validation_blocks: validation blocks for timeseries data,

    :return Iterator[Tuple[Union[InputData, MultiModalData],
                           Union[InputData, MultiModalData]]]: return split train/test data
    """

    # Define base class for generate cv folds
    retain_first_target = False
    if data.task.task_type is TaskTypesEnum.ts_forecasting:
        horizon = data.task.task_params.forecast_length * validation_blocks
        kf = TsInputDataSplit(n_splits=cv_folds, test_size=horizon)
        # for multi_ts use first target column as main target
        retain_first_target = True
    elif data.task.task_type is TaskTypesEnum.classification and stratify:
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    else:
        kf = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_seed)

    # Split
    for train_ids, test_ids in kf.split(data.target, data.target):
        train_data = _split_input_data_by_indexes(data, train_ids)
        test_data = _split_input_data_by_indexes(data, test_ids, retain_first_target=retain_first_target)
        yield train_data, test_data
