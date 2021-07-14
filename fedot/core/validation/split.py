from typing import Iterator, Tuple, Optional

import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup


class OneFoldSplit:
    """ Perform one fold split (hold out) for InputData structures """

    def __init__(self):
        pass

    @staticmethod
    def input_split(input_data: InputData, folds: Optional[int] = None):
        # Train test split
        train_input, test_input = train_test_data_setup(input_data)

        yield train_input, test_input


class TsSplit(TimeSeriesSplit):
    """ Perform time series splitting for cross validation """

    def __init__(self, validation_blocks: int, **params):
        super().__init__(**params)
        self.validation_blocks = validation_blocks
        self.params = params

    def input_split(self, input_data: InputData, folds: int = None) -> Iterator[Tuple[InputData, InputData]]:
        """ Splitting into datasets for train and validation using
        "in-sample forecasting" algorithm

        :param input_data: InputData for splitting
        :param folds: number of folds, which will be used for validation
        """
        # Transform InputData into numpy array
        data_for_split = np.array(input_data.target)
        # Set appropriate validation_blocks number in supplementary info
        input_data.supplementary_data.validation_blocks = self.validation_blocks

        for train_ids, test_ids in super().split(data_for_split):
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
                               data_type=data.data_type,
                               supplementary_data=data.supplementary_data)
        test_data = InputData(idx=idx_for_test,
                              features=test_features,
                              target=test_target,
                              task=data.task,
                              data_type=data.data_type,
                              supplementary_data=data.supplementary_data)

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

    try:
        tscv = TsSplit(gap=0, validation_blocks=validation_blocks,
                       n_splits=folds, test_size=horizon)
        for train_data, test_data in tscv.input_split(data, folds):
            yield train_data, test_data
    except ValueError:
        log.info(f'Time series length too small for cross validation with {folds} folds. Perform one fold validation')
        # Perform one fold validation (folds parameter will be ignored)
        one_fold_split = OneFoldSplit()
        data.supplementary_data.validation_blocks = None
        for train_data, test_data in one_fold_split.input_split(data, folds):
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

    return features, target
