from typing import Optional

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from fedot.core.chains.chain_ts_wrappers import in_sample_ts_forecast
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.validation.tune.simple import fit_predict_one_fold
from fedot.core.validation.tune.time_series import _calculate_n_splits, _choose_several_folds


def _in_sample_ts_validation(chain, data, validation_blocks):
    """ In-sample forecasting on several validations blocks is provided

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
                            data_type=DataTypesEnum.ts,
                            supplementary_data=data.supplementary_data)

    chain.fit_from_scratch(train_input)
    predicted_values = in_sample_ts_forecast(chain=chain,
                                             input_data=data,
                                             horizon=horizon)
    return test_target, predicted_values


def ts_cross_validation_tuning(chain, reference_data: InputData, log, validation_blocks):
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
        if log is not None:
            message = 'Time series length too small for cross validation. Perform one fold validation'
            log.info(message)
        actual_values, predicted_values = fit_predict_one_fold(reference_data, chain)
        predicted_values = np.ravel(np.array(predicted_values))
        actual_values = np.ravel(actual_values)

        return actual_values, predicted_values

    i = 0
    actual_values = []
    predicted_values = []
    for train_ids, test_ids in tscv.split(reference_data.features):
        if i in folds_to_use:
            actual, pred = perform_ts_validation(chain, reference_data, train_ids,
                                                  test_ids, validation_blocks)
            # Add actual and predicted values into common holder
            actual_values.extend(list(actual))
            predicted_values.extend(list(pred))
        i += 1

    actual_values = np.ravel(np.array(actual_values))
    predicted_values = np.ravel(np.array(predicted_values))

    return actual_values, predicted_values


def perform_ts_validation(chain, data, train_ids, test_ids, validation_blocks):
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

    test_target, predicted_values = _in_sample_ts_validation(chain, updated_input, validation_blocks)
    return test_target, predicted_values


def in_sample_composer_validation(chain, data, validation_blocks=3):
    horizon = data.task.task_params.forecast_length * validation_blocks
    predicted_values = in_sample_ts_forecast(chain=chain,
                                             input_data=data,
                                             horizon=horizon)

    # Clip actu
    actual_values = data.target[-horizon:]
    return actual_values, predicted_values
