import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.chains.chain_ts_wrappers import in_sample_ts_forecast
from fedot.core.data.data_split import train_test_data_setup

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


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


def ts_cross_validation(chain, data):
    """ Provide K-fold cross validation for time series with using in-sample
    forecasting on each step (fold)
    """
    # Forecast horizon for each fold
    validation_blocks = 2
    horizon = data.task.task_params.forecast_length * validation_blocks

    # Estimate appropriate number of splits
    n_splits = _calculate_n_splits(data, horizon)
    # Get numbers of folds for validation
    folds_to_use = _choose_several_folds(n_splits)

    try:
        tscv = TimeSeriesSplit(gap=0, n_splits=n_splits,
                               test_size=horizon)
    except ValueError:
        raise ValueError('Time series length too small for cross validation')

    i = 0
    actual_values = []
    predicted_values = []
    metrics = []
    for train_ids, test_ids in tscv.split(data.features):
        if i in folds_to_use:
            actual, pred = perform_ts_validation(chain, data, train_ids, test_ids,
                                                 validation_blocks)
            # TODO удалить блок с визуализацией
            ###################################
            len_train = len(train_ids)
            len_test = len(test_ids)
            if i == 1:
                label_1, label_2 = 'Actual values', 'Predicted'
            else:
                label_1, label_2 = None, None
            plt.plot(np.arange(0, len_train), data.features[train_ids], c='green', label=label_1)
            plt.plot(np.arange(len_train, len_train + len_test), actual, c='green')
            plt.plot(np.arange(len_train, len_train + len_test), pred, c='blue', label=label_2)
            ###################################

            # Add actual and predicted values into common holder
            actual_values.extend(list(actual))
            predicted_values.extend(list(pred))
            metric = mean_squared_error(actual, pred, squared=False)
            print(f'Значение метрики для CV {i} - {metric}')
            metrics.append(metric)
        i += 1

    print(f' A + B + C = {np.array(metrics).mean()}')
    actual_values = np.ravel(np.array(actual_values))
    predicted_values = np.ravel(np.array(predicted_values))
    print('------------------------------------------------------------------')
    print(f'Усредненное значение метрики: {mean_squared_error(actual_values, predicted_values, squared=False)}')
    print('------------------------------------------------------------------')
    # TODO удалить блок с визуализацией
    ###################################
    plt.grid()
    plt.legend()
    metrics = np.array(metrics)
    plt.title(f'Metric values: {np.round_(metrics, decimals=2)}, mean: {mean_squared_error(actual_values, predicted_values, squared=False):.2f}')
    plt.close()
    ###################################
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

    test_target, predicted_values = in_sample_ts_validation(chain, updated_input, validation_blocks)
    return test_target, predicted_values


def _calculate_n_splits(data, horizon: int):
    """ Calculated number of splits which will not lead to the errors """

    n_splits = len(data.features) // horizon
    # Remove two splits to allow algorithm get more data for train
    n_splits = n_splits - 1
    return n_splits


def _choose_several_folds(n_splits):
    """ Choose ids of several folds for further testing """

    # If there not enough folds in time series - take all of them
    if n_splits < 3:
        return np.arange(0, n_splits)
    else:
        # Randomly choose subsample of folds
        smallest_part = 1
        medium_part = int(round(n_splits/2))
        biggest_part = n_splits - 1
        return np.array([smallest_part])
