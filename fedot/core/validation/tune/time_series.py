import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.chains.chain_ts_wrappers import in_sample_ts_forecast
from fedot.core.validation.tune.simple import fit_predict_one_fold


def in_sample_ts_validation(chain, data, validation_blocks: int = 3):
    """ In-sample forecasting on three validations blocks is provided

    :param chain: Chain to validate
    :param data: InputData for validation
    :param validation_blocks: number of validation blocks
    """
    # Define forecast horizon for validation
    horizon = data.task.task_params.forecast_length * validation_blocks

    # Divide into train and test
    y_train_part = data.target[:-horizon]
    x_train_part = data.features[:-horizon]

    # Target length is equal to the forecast horizon
    test_target = np.ravel(data.target[-horizon:])

    # InputData for train
    train_input = InputData(idx=range(0, len(y_train_part)), features=x_train_part,
                            target=y_train_part, task=data.task,
                            data_type=DataTypesEnum.ts)

    chain.fit_from_scratch(train_input)
    predicted_values = in_sample_ts_forecast(chain=chain,
                                             input_data=data,
                                             horizon=horizon)
    return test_target, predicted_values


def ts_cross_validation(reference_data: InputData, chain, log, validation_blocks: int = 3):
    """ Provide K-fold cross validation for time series with using in-sample
    forecasting on each step (fold)
    """
    # Forecast horizon for each fold
    horizon = reference_data.task.task_params.forecast_length * validation_blocks

    # Estimate appropriate number of splits
    n_splits = _calculate_n_splits(reference_data, horizon)
    # Get numbers of folds for validation
    folds_to_use = _choose_several_folds(n_splits)

    try:
        tscv = TimeSeriesSplit(gap=0, n_splits=n_splits,
                               test_size=horizon)
    except ValueError:
        log.info('Time series length too small for cross validation. Perform one fold validation')
        actual_values, predicted_values = fit_predict_one_fold(reference_data, chain)
        predicted_values = np.ravel(np.array(predicted_values))
        actual_values = np.ravel(actual_values)

        return actual_values, predicted_values

    i = 0
    actual_values = []
    predicted_values = []
    for train_ids, test_ids in tscv.split(reference_data.features):
        if i in folds_to_use:
            actual, pred = _perform_ts_validation(chain, reference_data, train_ids,
                                                  test_ids, validation_blocks)
            # Add actual and predicted values into common holder
            actual_values.extend(list(actual))
            predicted_values.extend(list(pred))
        i += 1

    actual_values = np.ravel(np.array(actual_values))
    predicted_values = np.ravel(np.array(predicted_values))

    return actual_values, predicted_values


def _perform_ts_validation(chain, data, train_ids, test_ids, validation_blocks):
    """ Time series in-sample forecast evaluation

    :param chain: chain to validate
    :param data: InputData
    :param train_ids: list with indices of train elements
    :param test_ids: list with indices of test elements
    :param validation_blocks: number of blocks for validation
    """
    ids_for_validation = np.hstack((train_ids, test_ids))

    # Generate new InputData
    features_crop = data.features[ids_for_validation]
    target_crop = data.target[ids_for_validation]
    updated_input = InputData(idx=range(0, len(target_crop)), features=features_crop,
                              target=target_crop, task=data.task, data_type=DataTypesEnum.ts)

    test_target, predicted_values = in_sample_ts_validation(chain, updated_input, validation_blocks)
    return test_target, predicted_values


def _calculate_n_splits(data, horizon: int):
    """ Calculated number of splits which will not lead to the errors """

    n_splits = len(data.features) // horizon
    # Remove one split to allow algorithm get more data for train
    n_splits = n_splits - 1
    return n_splits


def _choose_several_folds(n_splits):
    """ Choose ids of several folds for further testing """

    # If there not enough folds in time series - take all of them
    if n_splits < 3:
        return np.arange(0, n_splits)
    else:
        # Choose last folds for validation
        biggest_part = n_splits - 1
        medium_part = biggest_part - 1
        smallest_part = medium_part - 1
        return np.array([smallest_part, medium_part, biggest_part])
