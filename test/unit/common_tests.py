from copy import deepcopy

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.utils import set_random_seed


def is_predict_ignores_target(predict_func, input_data, data_arg_name, predict_args=None):
    set_random_seed(0)

    if predict_args is None:
        predict_args = {}

    if isinstance(input_data, str):
        input_data = InputData.from_csv(input_data)

    input_data_without_target = deepcopy(input_data)
    input_data_without_target.target = None
    predictions = predict_func(**{data_arg_name: input_data}, **predict_args)
    predictions_without_target = predict_func(**{data_arg_name: input_data_without_target}, **predict_args)

    if isinstance(predictions, OutputData):
        pred_values = predictions.predict
        pred_values_without_target = predictions_without_target.predict
    else:
        pred_values = predictions
        pred_values_without_target = predictions_without_target

    return np.equal(pred_values, pred_values_without_target).all()
