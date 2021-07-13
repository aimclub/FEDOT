import numpy as np

from fedot.core.chains.chain_ts_wrappers import in_sample_ts_forecast
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.validation.split import ts_cv_generator


def cross_validation_predictions(chain, reference_data: InputData, log, cv_folds: int):
    """ Provide K-fold cross validation for time series with using in-sample
    forecasting on each step (fold)
    """

    # Place where predictions and actual values will be loaded
    predictions = []
    targets = []
    for _, test_data in ts_cv_generator(reference_data, log, cv_folds):
        if test_data.supplementary_data.validation_blocks is None:
            # One fold validation
            output_pred = chain.predict(test_data)
            predictions = output_pred.predict
            targets = output_pred.target
            break
        else:
            # Cross validation: get number of validation blocks per each fold
            validation_blocks = test_data.supplementary_data.validation_blocks
            horizon = test_data.task.task_params.forecast_length * validation_blocks

            predicted_values = in_sample_ts_forecast(chain=chain,
                                                     input_data=test_data,
                                                     horizon=horizon)
            # Clip actual data by the forecast horizon length
            actual_values = test_data.target[-horizon:]
            predictions.extend(predicted_values)
            targets.extend(actual_values)

    predictions, targets = np.ravel(np.array(predictions)), np.ravel(np.array(targets))
    return predictions, targets


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
    updated_input = InputData(idx=np.arange(0, len(target_crop)), features=features_crop,
                              target=target_crop, task=data.task, data_type=DataTypesEnum.ts)

    test_target, predicted_values = _in_sample_ts_validation(chain, updated_input, validation_blocks)
    return test_target, predicted_values


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
    train_input = InputData(idx=np.arange(0, len(y_train_part)), features=x_train_part,
                            target=y_train_part, task=data.task,
                            data_type=DataTypesEnum.ts,
                            supplementary_data=data.supplementary_data)

    chain.fit_from_scratch(train_input)
    predicted_values = in_sample_ts_forecast(chain=chain,
                                             input_data=data,
                                             horizon=horizon)
    return test_target, predicted_values
