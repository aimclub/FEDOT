from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup


class OneFoldSplit:
    """ Perform one fold split (hold out) for InputData structures """

    def __init__(self):
        pass

    @staticmethod
    def input_split(input_data: InputData, folds: int = None):
        # Train test split
        train_input, test_input = train_test_data_setup(input_data)

        yield train_input, test_input


class TsSplit(TimeSeriesSplit):
    """ Perform time series splitting for cross validation """

    def __init__(self, **params):
        super().__init__(**params)
        self.params = params

    def input_split(self, input_data: InputData, folds: int = None):
        # Transform InputData into numpy array
        data_for_split = np.array(input_data.target)

        # Get numbers of folds for validation
        folds_to_use = _choose_several_folds(self.param['n_splits'], folds)

        i = 0
        for train_ids, test_ids in super(**self.params).split(data_for_split):
            if i in folds_to_use:
                # Return train part by ids
                features, target = _ts_data_by_index(train_ids, test_ids, input_data)
                validation_data = InputData(idx=range(0, len(target)),
                                            features=features, target=target,
                                            task=input_data.task,
                                            data_type=input_data.data_type)
                yield validation_data, None
            i += 1


def tabular_cv_generator(data: InputData, folds: int) -> Iterator[Tuple[InputData, InputData]]:
    """ The function for splitting data into a train and test samples
        in the InputData format for KFolds cross validation. The function
        return a generator of tuples, consisting of a pair of train, test.

    :param data: InputData for train and test splitting
    :param folds: number of folds

    :return Iterator[InputData, InputData]: return split train/test data
    """
    kf = KFold(n_splits=folds)

    for train_idxs, test_idxs in kf.split(data.features):
        train_features, train_target = _table_data_by_index(train_idxs, data)
        test_features, test_target = _table_data_by_index(test_idxs, data)

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


def ts_cv_generator(data, log, folds: int, validation_blocks: int = 3) -> Iterator[Tuple[InputData, InputData]]:
    """ Splitting data for time series cross validation

    :param data: source InputData with time series data type
    :param log: log object
    :param folds: number of folds
    :param validation_blocks: number of validation block per each fold
    """
    # Forecast horizon for each fold
    horizon = data.task.task_params.forecast_length * validation_blocks

    # Estimate appropriate number of splits
    n_splits = _calculate_n_splits(data, horizon)

    try:
        tscv = TsSplit(gap=0, n_splits=n_splits, test_size=horizon)
    except ValueError:
        log.info('Time series length too small for cross validation. Perform one fold validation')
        # Perform one fold validation (folds parameter will be ignored)
        tscv = OneFoldSplit()

    for train_data, test_data in tscv.input_split(data, folds):
        yield train_data, test_data


def _table_data_by_index(index, values: InputData):
    features = values.features[index, :]
    target = np.take(values.target, index)

    return features, target


def _ts_data_by_index(train_ids, test_ids, data):
    ids_for_validation = np.hstack((train_ids, test_ids))
    # Crop features and target
    features = data.features[ids_for_validation]
    target = data.target[ids_for_validation]

    return features, target


def _calculate_n_splits(data, horizon: int):
    """ Calculated number of splits which will not lead to the errors
    for time series cross validation
    """

    n_splits = len(data.features) // horizon
    # Remove one split to allow algorithm get more data for train
    n_splits = n_splits - 1
    return n_splits


def _choose_several_folds(n_splits, folds):
    """ Choose ids of several folds for further testing for time
    series cross validation

    :param n_splits: number of all splits
    :param folds: number of splits to use
    """

    # If there not enough folds in time series - take all of them
    if n_splits < folds:
        return np.arange(0, n_splits)
    else:
        # Choose last folds for validation
        # Start with biggest part (last fold)
        current_biggest_part = n_splits - 1
        parts_list = [current_biggest_part]

        for fold_id in range(len(parts_list)):
            current_biggest_part -= 1
            parts_list.append(current_biggest_part)
        return np.array(parts_list)
