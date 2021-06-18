import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.chains.chain_ts_wrappers import in_sample_ts_forecast
from fedot.core.data.data_split import train_test_data_setup


def fit_predict_one_fold(chain, data):
    """ Simple strategy for model evaluation based on one folder check

    :param chain: Chain to validate
    :param data: InputData for validation
    """

    # Train test split
    train_input, predict_input = train_test_data_setup(data)
    test_target = np.array(predict_input.target)

    chain.fit_from_scratch(train_input)
    predicted_output = chain.predict(predict_input)
    predictions = np.array(predicted_output.predict)

    return test_target, predictions


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


def ts_cross_validation(chain, data, cv: int = 10):
    """ Provide K-fold cross validation for time series with using in-sample
    forecasting on each step (fold)
    """
    # Forecast horizon for each fold
    validation_blocks = 2
    horizon = data.task.task_params.forecast_length * validation_blocks

    # Estimate appropriate number of splits
    n_splits = _calculate_n_splits(data, horizon)
    # Get numbers of folds for validation
    folds_to_use = _choose_random_folds(n_splits, cv)

    tscv = TimeSeriesSplit(gap=0, n_splits=n_splits,
                           test_size=horizon)

    i = 0
    actual_values = []
    predicted_values = []
    for train_ids, test_ids in tscv.split(data.features):
        if i in folds_to_use:
            actual, pred = perform_ts_validation(chain, data, train_ids, test_ids,
                                                 validation_blocks)
            # Add actual and predicted values into common holder
            actual_values.extend(list(actual))
            predicted_values.extend(list(pred))
        i += 1
    actual_values = np.ravel(np.array(actual_values))
    predicted_values = np.ravel(np.array(predicted_values))
    return actual_values, predicted_values


def perform_ts_validation(chain, data, train_ids, test_ids, validation_blocks):
    """ Time series in-sample forecast evaluation """
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
    # Remove two splits to allow algorithm get more data for train
    n_splits = n_splits - 2
    return n_splits


def _choose_random_folds(n_splits, cv: int):
    """ Choose ids of several folds for further testing """

    # If there not enough folds in time series - take all of them
    if cv >= n_splits:
        return np.arange(0, n_splits)
    else:
        # Randomly choose subsample of folds
        fold_ids = [3, n_splits]
        return np.random.choice(fold_ids, cv)
