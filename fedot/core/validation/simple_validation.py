import numpy as np

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.chains.chain_ts_wrappers import in_sample_ts_forecast
from fedot.core.data.data_split import train_test_data_setup


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
