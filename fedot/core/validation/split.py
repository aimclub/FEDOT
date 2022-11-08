from typing import Iterator, Optional, Tuple, Type

import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.model_selection._split import _BaseKFold

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import LoggerAdapter, default_log
from fedot.core.repository.dataset_types import DataTypesEnum


class OneFoldInputDataSplit:
    """ Perform one fold split (hold out) for InputData structures """

    def __init__(self):
        pass

    @staticmethod
    def input_split(input_data: InputData, **kwargs):
        # Train test split
        train_input, test_input = train_test_data_setup(input_data, **kwargs)

        yield train_input, test_input


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

    def __init__(self, validation_blocks: int, **params):
        super().__init__(**params)
        self.validation_blocks = validation_blocks
        self.params = params

    def input_split(self, input_data: InputData) -> Iterator[Tuple[InputData, InputData]]:
        """ Splitting into datasets for train and validation using
        "in-sample forecasting" algorithm

        :param input_data: InputData for splitting
        """
        # Transform InputData into numpy array
        data_for_split = np.array(input_data.target)

        for train_ids, test_ids in super().split(data_for_split):
            if len(train_ids) <= len(test_ids):
                raise ValueError("Train size will be too small with selected number of folds and validation blocks")
            # Return train part by ids
            train_features, train_target = _ts_data_by_index(train_ids, train_ids, input_data)
            train_data = InputData(idx=np.arange(0, len(train_target)),
                                   features=train_features, target=train_target,
                                   task=input_data.task,
                                   data_type=input_data.data_type,
                                   supplementary_data=input_data.supplementary_data)

            # Unit all ids for "in-sample validation"
            all_ids = np.hstack((train_ids, test_ids))
            # In-sample validation dataset
            val_features, val_target = _ts_data_by_index(all_ids, all_ids, input_data)
            validation_data = InputData(idx=np.arange(0, len(val_target)),
                                        features=val_features, target=val_target,
                                        task=input_data.task,
                                        data_type=input_data.data_type,
                                        supplementary_data=input_data.supplementary_data)

            yield train_data, validation_data


def tabular_cv_generator(data: InputData,
                         folds: int,
                         split_method: Type[_BaseKFold] = KFold) -> Iterator[Tuple[InputData, InputData]]:
    """ The function for splitting data into a train and test samples
        in the InputData format for KFolds cross validation. The function
        return a generator of tuples, consisting of a pair of train, test.

    :param data: InputData for train and test splitting
    :param folds: number of folds
    :param split_method: method to split data (f.e. stratify KFold)

    :return Iterator[InputData, InputData]: return split train/test data
    """
    kf = split_method(n_splits=folds, shuffle=True, random_state=42)

    for train_idxs, test_idxs in kf.split(data.features, data.target):
        train_features, train_target = _table_data_by_index(train_idxs, data)
        test_features, test_target = _table_data_by_index(test_idxs, data)

        idx_for_train = np.arange(0, len(train_features))
        idx_for_test = np.arange(0, len(test_features))

        train_data = InputData(idx=idx_for_train,
                               features=train_features,
                               target=train_target,
                               task=data.task,
                               data_type=data.data_type,
                               supplementary_data=data.supplementary_data)
        test_data = InputData(idx=idx_for_test,
                              features=test_features,
                              target=test_target,
                              task=data.task,
                              data_type=data.data_type,
                              supplementary_data=data.supplementary_data)

        yield train_data, test_data


def ts_cv_generator(data: InputData, folds: int,
                    validation_blocks: int = 1, log: Optional[LoggerAdapter] = None) \
        -> Iterator[Tuple[InputData, InputData]]:
    """ Splitting data for time series cross validation

    :param data: source InputData with time series data type
    :param folds: number of folds
    :param validation_blocks: number of validation block per each fold
    :param log: log object
    """
    if not log:
        log = default_log(prefix=__name__)
    validation_blocks = int(validation_blocks)
    # Forecast horizon for each fold
    horizon = data.task.task_params.forecast_length * validation_blocks

    try:
        tscv = TsInputDataSplit(gap=0, validation_blocks=validation_blocks,
                                n_splits=folds, test_size=horizon)
        for train_data, test_data in tscv.input_split(data):
            yield train_data, test_data
    except ValueError:
        log.info(f'Time series length too small for cross validation with {folds} folds. Perform one fold validation')
        # Perform one fold validation (folds parameter will be ignored)

        one_fold_split = OneFoldInputDataSplit()
        for train_data, test_data in one_fold_split.input_split(data, validation_blocks=validation_blocks):
            yield train_data, test_data


def _table_data_by_index(index, values: InputData):
    """ Allow to get tabular data by indexes of elements """
    features = values.features[index, :]
    target = np.take(values.target, index)

    return features, target


def _ts_data_by_index(train_ids, test_ids, data):
    """ Allow to get time series data by indexes of elements """
    features = data.features[train_ids]
    target = data.target[test_ids]

    # Use only the first time-series as target for multi_ts
    if data.data_type == DataTypesEnum.multi_ts:
        target = target[:, 0]

    return features, target
