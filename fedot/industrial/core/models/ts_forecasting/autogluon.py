# import shutil
# from copy import deepcopy
#
# import pandas as pd
# from fedot.core.data.data import InputData
#
# import autogluon as ag
# from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
#
# import logging
#
# logging.raiseExceptions = False
#
#
# class AutoGluonForecaster:
#     """
#     Class for time series forecasting with Autogluon
#     Source code: https://github.com/autogluon/autogluon
#     """
#
#     def __init__(self, **params):
#         super().__init__(**params)
#         self.target = 'value'
#         self.timeout = params.get('timeout', 60)
#         self.presets = params.get('presets')
#         self.model = None
#
#     def fit(self, input_data: InputData):
#         self.model = self._init_model(input_data.task.task_params.forecast_length)
#         historical_values = deepcopy(input_data.features)
#         historical_values['idx'] = '0'
#         train_data = TimeSeriesDataFrame.from_data_frame(
#             historical_values,
#             id_column="idx",
#             timestamp_column="datetime"
#         )
#
#         self.model.fit(
#             train_data,
#             presets=self.presets,
#             time_limit=self.timeout,
#         )
#
#     def predict(self, historical_values: pd.DataFrame, **kwargs):
#         """ Use obtained model to make predictions """
#         # Update model weights
#         historical_values = historical_values.copy()
#         historical_values['idx'] = '0'
#         train_data = TimeSeriesDataFrame.from_data_frame(
#             historical_values,
#             id_column="idx",
#             timestamp_column="datetime"
#         )
#         predictions = self.model.predict(train_data)
#         predictions.head()
#         shutil.rmtree('AutogluonModels')
#         return predictions
#
#     def _init_model(self, forecast_horizon):
#         return TimeSeriesPredictor(prediction_length=forecast_horizon, target="value", eval_metric="sMAPE")
