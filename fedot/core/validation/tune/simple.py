import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup


def fit_predict_one_fold(data: InputData, pipeline) -> [InputData, OutputData]:
    """ Simple strategy for model evaluation based on one folder check

    :param data: InputData for validation
    :param pipeline: Pipeline to validate
    """

    # Train test split
    train_input, predict_input = train_test_data_setup(data)

    pipeline.fit_from_scratch(train_input)
    predicted_output = pipeline.predict(predict_input)

    return predict_input, predicted_output
