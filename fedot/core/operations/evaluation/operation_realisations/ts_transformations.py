from copy import copy
from typing import List, Optional

import pandas as pd
import numpy as np

from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.operations.evaluation.\
    operation_realisations.abs_interfaces import OperationRealisation


class LaggedTransformation(OperationRealisation):
    """ Realisation of lagged transformation for time series forecasting"""

    def __init__(self, **params: Optional[dict]):
        super().__init__()

        if not params:
            self.window_size = 10
        else:
            self.window_size = params.get('window_size')

    def fit(self, input_data):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data, is_fit_chain_stage: bool):
        """ Method for transformation of time series to lagged form

        :param input_data: data with features, target and ids to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return output_data: output data with transformed features table
        """
        parameters = input_data.task.task_params
        old_idx = input_data.idx

        if is_fit_chain_stage:
            # Transformation for fit stage of the chain
            target = input_data.target
            features = input_data.features
            # Prepare features for training
            new_idx, features_columns = self._prepare_features(idx=old_idx,
                                                               features=features,
                                                               window_size=self.window_size)

            # Transform target
            new_idx, features_columns, new_target = self._prepare_target(idx=new_idx,
                                                                         features_columns=features_columns,
                                                                         target=target,
                                                                         parameters=parameters)
            # Update target for Input Data
            input_data.target = new_target
        else:
            # Transformation for predict stage of the chain
            features = np.array(input_data.features)
            features_columns = features[-self.window_size:]
            features_columns = features_columns.reshape(1, -1)
            new_idx = old_idx[-self.window_size:]

        # Update idx and features
        input_data.idx = new_idx
        output_data = self._convert_to_output(input_data,
                                              features_columns)
        return output_data

    @staticmethod
    def _prepare_features(idx, features, window_size):
        """ Method convert time series to lagged form. Transformation applied
        only for generating features table.

        :param idx: the indices of the time series to convert
        :param features: source time series
        :param window_size: size of sliding window, which defines lag

        :return updated_idx: clipped indices of time series
        :return features_columns: lagged time series feature table
        """

        # Convert data to lagged form
        lagged_dataframe = pd.DataFrame({'target': features})
        vals = lagged_dataframe['target']
        for i in range(1, window_size + 1):
            frames = [lagged_dataframe, vals.shift(i)]
            lagged_dataframe = pd.concat(frames, axis=1)

        # Remove incomplete rows
        lagged_dataframe.dropna(inplace=True)

        transformed = np.array(lagged_dataframe)

        # Generate dataset with features
        features_columns = transformed[:, 1:]
        features_columns = np.fliplr(features_columns)

        # First n elements in time series are removed
        updated_idx = idx[window_size:]

        return updated_idx, features_columns

    @staticmethod
    def _prepare_target(idx, features_columns, target, parameters):
        """ Method convert time series to lagged form. Transformation applied
        only for generating target table (time series considering as multi-target
        regression task).

        :param idx: remaining indices after lagged feature table generation
        :param features_columns: lagged feature table
        :param target: source time series
        :param parameters: parameters of the task

        :return updated_idx: clipped indices of time series
        :return updated_features: clipped lagged feature table
        :return updated_target: lagged target table
        """

        # Update target (clip first "window size" values)
        ts_target = target[idx]

        # Multi-target transformation
        if parameters.forecast_length > 1:
            # Target transformation
            df = pd.DataFrame({'target': ts_target})
            vals = df['target']
            for i in range(1, parameters.forecast_length):
                frames = [df, vals.shift(-i)]
                df = pd.concat(frames, axis=1)

            # Remove incomplete rows
            df.dropna(inplace=True)
            updated_target = np.array(df)

            threshold = -parameters.forecast_length + 1
            updated_idx = idx[: threshold]
            updated_features = features_columns[: threshold]
        else:
            updated_idx = idx
            updated_features = features_columns
            updated_target = ts_target

        return updated_idx, updated_features, updated_target

    def get_params(self):
        raise NotImplementedError()


