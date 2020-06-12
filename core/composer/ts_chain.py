import numpy as np

from core.composer.chain import Chain
from core.models.data import InputData
from core.repository.tasks import TaskTypesEnum


class TsForecastingChain(Chain):
    def forecast(self, initial_data: InputData, supplementary_data: InputData):
        if not self.is_all_cache_actual():
            raise ValueError('Chain for the time series forecasting was not fitted yet.')

        if supplementary_data.task.task_type is not TaskTypesEnum.ts_forecasting:
            raise ValueError('TsForecastingChain can be used for the ts_forecasting task only.')

        # predict_data contains task description and additional (exogenous) variables
        forecast_length = supplementary_data.task.task_params.forecast_length

        # check if predict features contains additional (exogenous) variables
        with_exog = (supplementary_data.features is not None and
                     not np.array_equal(supplementary_data.features, supplementary_data.target))

        # initial data for the first prediction
        pre_history_start = len(initial_data.idx) - initial_data.task.task_params.max_window_size
        pre_history_end = len(initial_data.idx) + 1
        data_for_forecast = initial_data.subset(start=pre_history_start, end=pre_history_end)

        full_prediction = []
        forecast_steps_num = int(np.ceil(len(supplementary_data.idx) / forecast_length))
        for forecast_step in range(forecast_steps_num):
            # prediction for forecast_length steps
            stepwise_prediction = self.predict(data_for_forecast).predict
            if len(stepwise_prediction.shape) > 1:
                # multi-step prediction
                stepwise_prediction = stepwise_prediction[-1, :]
                full_prediction.extend(stepwise_prediction)
            else:
                # single-step prediction
                stepwise_prediction = stepwise_prediction[-1]
                full_prediction.append(stepwise_prediction)

            # add additional variable from external source
            if with_exog:
                data_for_forecast = _prepare_exog_features(data_for_forecast, supplementary_data,
                                                           stepwise_prediction,
                                                           forecast_step, forecast_length)
            else:
                predicted_ts = np.append(data_for_forecast.target, stepwise_prediction)
                data_for_forecast.target = np.stack(predicted_ts)
                data_for_forecast.features = data_for_forecast.target

            # extend idx for prediction
            if forecast_length > 1:
                indices_for_forecast = (data_for_forecast.idx[-1] +
                                        list(range(1, len(stepwise_prediction) + 1)))

                data_for_forecast.idx = np.append(data_for_forecast.idx,
                                                  indices_for_forecast)
            else:
                data_for_forecast.idx = np.append(data_for_forecast.idx,
                                                  data_for_forecast.idx[-1] + 1)

        return full_prediction


def _prepare_exog_features(data_for_prediction: InputData,
                           exog_data: InputData,
                           last_prediction: np.array,
                           forecast_step: int, forecast_length: int) -> InputData:
    new_features = []
    if len(data_for_prediction.features.shape) == 1:
        # if one exog feature
        exog_features_num = 1
    else:
        # if several exog features
        exog_features_num = data_for_prediction.features.shape[1]

    new_part_len = 0
    if exog_features_num > 1:
        for exog_feat_id in range(exog_features_num):
            exog_feature = data_for_prediction.features[:, exog_feat_id]
            new_exog_values = \
                exog_data.features[forecast_step * forecast_length:
                                   ((forecast_step + 1) * forecast_length), exog_feat_id]
            new_feature = np.append(exog_feature, new_exog_values)
            new_features.append(new_feature)
            new_part_len = len(new_features[0])
    else:
        exog_feature = data_for_prediction.features
        new_exog_values = \
            exog_data.features[forecast_step * forecast_length:
                               ((forecast_step + 1) * forecast_length)]
        new_features = np.append(exog_feature, new_exog_values)
        new_part_len = len(new_features)

    # add predicted time series to features for next prediction
    predicted_ts = np.append(data_for_prediction.target,
                             last_prediction)

    # cut the prediction if it's too long (actual for the last forecast step)
    predicted_ts = predicted_ts[0: new_part_len]
    # new_features.append(predicted_ts)
    data_for_prediction.target = predicted_ts
    data_for_prediction.features = np.stack(np.asarray(new_features)).T

    return data_for_prediction
