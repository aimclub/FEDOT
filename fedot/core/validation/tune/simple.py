import numpy as np

from fedot.core.data.data_split import train_test_data_setup


def fit_predict_one_fold(data, chain):
    """ Simple strategy for model evaluation based on one folder check

    :param data: InputData for validation
    :param chain: Chain to validate
    """

    # Train test split
    train_input, predict_input = train_test_data_setup(data)
    test_target = np.array(predict_input.target)

    chain.fit_from_scratch(train_input)
    predicted_output = chain.predict(predict_input)
    predictions = np.array(predicted_output.predict)

    return test_target, predictions
