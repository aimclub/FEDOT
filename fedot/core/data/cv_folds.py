from typing import Iterator, Optional, Tuple, Union

import numpy as np

from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.tasks import TaskTypesEnum
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.model_selection._split import StratifiedKFold

from fedot.core.data.data import InputData


def cv_generator(data: Union[InputData, MultiModalData],
                 cv_folds: int,
                 shuffle: bool = False,
                 random_seed: int = 42,
                 stratify: bool = True,
                 validation_blocks: Optional[int] = None) -> Iterator[Tuple[Union[InputData, MultiModalData],
                                                                            Union[InputData, MultiModalData]]]:
    """ The function for splitting data into a train and test samples
        for cross validation. The function return a generator of tuples,
        consisting of a pair of train, test.

    :param data: data for train and test splitting
    :param shuffle: is data need shuffle
    :param cv_folds: number of folds
    :param random_seed: random seed for shuffle
    :param stratify: `True` to make stratified samples for classification task
    :param validation_blocks: validation blocks for timeseries data,

    :return Iterator[Tuple[Union[InputData, MultiModalData],
                           Union[InputData, MultiModalData]]]: return split train/test data
    """

    # Define base class for generate cv folds
    if data.task.task_type is TaskTypesEnum.ts_forecasting:
        horizon = data.task.task_params.forecast_length * validation_blocks
        kf = TimeSeriesSplit(n_splits=cv_folds, test_size=horizon)
    elif data.task.task_type is TaskTypesEnum.classification and stratify:
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    else:
        kf = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_seed)

    # Split
    for train_ids, test_ids in kf.split(data.target, data.target):
        train_data = data.slice_by_index(train_ids)
        test_data = data.slice_by_index(test_ids)
        # use first col in multi_ts ad target
        if (data.task.task_type is TaskTypesEnum.ts_forecasting and
                len(test_data.target.shape) > 1):
            test_data.target = test_data.target[:, 0]
        yield train_data, test_data
