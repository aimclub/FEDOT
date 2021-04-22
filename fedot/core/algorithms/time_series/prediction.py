import numpy as np
from sklearn.metrics import mean_squared_error as mse_metric

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TsForecastingParams


def multistep_prediction_to_ts(prediction):
    # choose the early steps of first prediction
    first_pred_part = prediction[0, :-1]
    # choose last forecasting step only for each prediction
    prediction = prediction[:, -1]
    return np.concatenate((first_pred_part, prediction))


def cut_future_prediction_part(prediction, task_params: TsForecastingParams):
    # TODO add multivariate

    length_of_cut = task_params.forecast_length
    return prediction[:-length_of_cut]


def preserve_prediction_length(prediction, expected_length: int):
    if len(prediction) < expected_length:
        zeros = expected_length - len(prediction)
        zeros = np.asarray([np.nan] * zeros)
        if len(prediction.shape) == 1:
            prediction = np.concatenate((zeros, prediction))
        else:
            prediction_steps = []
            for forecast_depth in range(prediction.shape[1]):
                prediction_steps.append(np.concatenate((zeros, prediction[:, forecast_depth])))
            prediction = np.stack(np.asarray(prediction_steps)).T
    return prediction


def post_process_forecasted_ts(prediction, input_data: 'InputData'):
    task = input_data.task
    expected_length = len(input_data.idx)

    data_type = input_data.data_type

    if data_type == DataTypesEnum.ts_lagged_table:
        # because one step is devoted to the feature vector for future prediction
        expected_length = expected_length - 1

    if len(prediction.shape) > 1:
        prediction = multistep_prediction_to_ts(prediction)

    if not task.task_params.make_future_prediction and \
            data_type in [DataTypesEnum.ts_lagged_table]:
        # cut unwanted out-of-sample prediction
        prediction = cut_future_prediction_part(prediction, task.task_params)

    prediction = preserve_prediction_length(prediction, expected_length)

    return prediction


def ts_mse(obs, pred) -> float:
    return mse_metric(y_true=obs[~np.isnan(pred)],
                      y_pred=pred[~np.isnan(pred)],
                      squared=False)
