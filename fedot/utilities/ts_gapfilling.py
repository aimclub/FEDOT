from copy import deepcopy

import numpy as np
from scipy import interpolate

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.log import Log, default_log


def series_has_gaps_check(gapfilling_method):
    """ Check is time series has gaps or not. Return source array, if not """

    def wrapper(self, input_data, *args, **kwargs):
        gap_ids = np.ravel(np.argwhere(input_data == self.gap_value))
        if len(gap_ids) == 0:
            self.log.info(f'Array does not contain values marked as gaps {self.gap_value}')
            return input_data
        else:
            self.log.debug(f'Array contain values marked as gaps {self.gap_value}. Start gap-filling')
            filled_array = gapfilling_method(self, input_data, *args, **kwargs)
            return filled_array

    return wrapper


class SimpleGapFiller:
    """
    Base class used for filling in the gaps in time series with simple methods.
    Methods from the SimpleGapFiller class can be used for comparison with more
    complex models in class ModelGapFiller

    :param gap_value: value, which identify gap elements in array
    """

    def __init__(self, gap_value: float = -100.0, log: Log = None):
        self.gap_value = gap_value
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    @series_has_gaps_check
    def linear_interpolation(self, input_data: np.array):
        """
        Method allows to restore missing values in an array
        using linear interpolation

        :param input_data: array with gaps
        :return: array without gaps
        """
        output_data = np.array(input_data)

        # Process first and last elements in time series
        output_data = self._fill_first_and_last_gaps(input_data, output_data)

        # The indices of the known elements
        non_nan = np.ravel(np.argwhere(output_data != self.gap_value))
        # All known elements in the array
        masked_array = output_data[non_nan]
        f_interploate = interpolate.interp1d(non_nan, masked_array)
        x = np.arange(0, len(output_data))
        output_data = f_interploate(x)
        return output_data

    @series_has_gaps_check
    def local_poly_approximation(self, input_data, degree: int = 2,
                                 n_neighbors: int = 5):
        """
        Method allows to restore missing values in an array
        using Savitzky-Golay filter

        :param input_data: array with gaps
        :param degree: degree of a polynomial function
        :param n_neighbors: number of neighboring known elements of the time
        series that the approximation is based on
        :return: array without gaps
        """
        output_data = np.array(input_data)

        i_gaps = np.ravel(np.argwhere(output_data == self.gap_value))

        # Iterately fill in the gaps in the time series
        for gap_index in i_gaps:
            # Indexes of known elements (updated at each iteration)
            i_known = np.argwhere(output_data != self.gap_value)
            i_known = np.ravel(i_known)

            # Based on the indexes we calculate how far from the gap
            # the known values are located
            id_distances = np.abs(i_known - gap_index)

            # Now we know the indices of the smallest values in the array,
            # so sort indexes
            sorted_idx = np.argsort(id_distances)
            nearest_values = []
            nearest_indices = []
            for i in sorted_idx[:n_neighbors]:
                time_index = i_known[i]
                nearest_values.append(output_data[time_index])
                nearest_indices.append(time_index)
            nearest_values = np.array(nearest_values)
            nearest_indices = np.array(nearest_indices)

            local_coefs = np.polyfit(nearest_indices, nearest_values, degree)
            est_value = np.polyval(local_coefs, gap_index)
            output_data[gap_index] = est_value

        return output_data

    @series_has_gaps_check
    def batch_poly_approximation(self, input_data, degree: int = 3,
                                 n_neighbors: int = 10):
        """
        Method allows to restore missing values in an array using
        batch polynomial approximations.
        Approximation is applied not for individual omissions, but for
        intervals of omitted values

        :param input_data: array with gaps
        :param degree: degree of a polynomial function
        :param n_neighbors: the number of neighboring known elements of
        time series that the approximation is based on
        :return: array without gaps
        """
        output_data = np.array(input_data)

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for gap in new_gap_list:
            # Find the center point of the gap
            center_index = int((gap[0] + gap[-1]) / 2)

            # Indexes of known elements (updated at each iteration)
            i_known = np.argwhere(output_data != self.gap_value)
            i_known = np.ravel(i_known)

            # Based on the indexes we calculate how far from the gap
            # the known values are located
            id_distances = np.abs(i_known - center_index)

            # Now we know the indices of the smallest values in the array,
            # so sort indexes
            sorted_idx = np.argsort(id_distances)

            # Nearest known values to the gap
            nearest_values = []
            # And their indexes
            nearest_indices = []
            for i in sorted_idx[:n_neighbors]:
                # Getting the index value for the series - output_data
                time_index = i_known[i]
                # Using this index, we get the value of each of the "neighbors"
                nearest_values.append(output_data[time_index])
                nearest_indices.append(time_index)
            nearest_values = np.array(nearest_values)
            nearest_indices = np.array(nearest_indices)

            # Local approximation by an n-th degree polynomial
            local_coefs = np.polyfit(nearest_indices, nearest_values, degree)

            # Estimate our interval according to the selected coefficients
            est_value = np.polyval(local_coefs, gap)
            output_data[gap] = est_value

        return output_data

    def _parse_gap_ids(self, gap_list: list) -> list:
        """
        Method allows parsing source array with gaps indexes

        :param gap_list: array with indexes of gaps in array
        :return: a list with separated gaps in continuous intervals
        """

        new_gap_list = []
        local_gaps = []
        for index, gap in enumerate(gap_list):
            if index == 0:
                local_gaps.append(gap)
            else:
                prev_gap = gap_list[index - 1]
                if gap - prev_gap > 1:
                    # There is a "gap" between gaps
                    new_gap_list.append(local_gaps)

                    local_gaps = []
                    local_gaps.append(gap)
                else:
                    local_gaps.append(gap)
        new_gap_list.append(local_gaps)

        return new_gap_list

    def _fill_first_and_last_gaps(self, input_data: np.array, output_data: np.array):
        """ Eliminate gaps, which place first or last index in time series """
        non_nan_ids = np.ravel(np.argwhere(output_data != self.gap_value))
        non_nan = output_data[non_nan_ids]
        if np.isclose(input_data[0], self.gap_value):
            # First element is a gap - replace with first known value
            self.log.info(f'First element in the array were replaced by first known value')
            output_data[0] = non_nan[0]
        if np.isclose(input_data[-1], self.gap_value):
            # Last element is a gap - last known value
            self.log.info(f'Last element in the array were replaced by last known value')
            output_data[-1] = non_nan[-1]

        return output_data


class ModelGapFiller(SimpleGapFiller):
    """
    Class used for filling in the gaps in time series

    :param gap_value: value, which mask gap elements in array
    :param pipeline: TsForecastingPipeline object for filling in the gaps
    """

    def __init__(self, gap_value, pipeline, log: Log = None):
        super().__init__(gap_value, log)
        self.pipeline = pipeline

        # At least 6 elements needed to train pipeline with lagged transformation
        self.min_train_ts_length = 6

    @series_has_gaps_check
    def forward_inverse_filling(self, input_data):
        """
        Method fills in the gaps in the input array using forward and inverse
        directions of predictions

        :param input_data: data with gaps to filling in the gaps in it
        :return: array without gaps
        """
        output_data = np.array(input_data)

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for batch_index in range(len(new_gap_list)):

            preds = []
            weights = []
            # Two predictions are generated for each gap - forward and backward
            for direction_function in [self._forward, self._inverse]:
                weights_list, predicted_list = direction_function(output_data,
                                                                  batch_index,
                                                                  new_gap_list)
                weights.append(weights_list)
                preds.append(predicted_list)

            preds = np.array(preds)
            weights = np.array(weights)
            result = np.average(preds, axis=0, weights=weights)

            gap = new_gap_list[batch_index]
            # Replace gaps in an array with prediction values
            output_data[gap] = result

        return output_data

    @series_has_gaps_check
    def forward_filling(self, input_data):
        """
        Method fills in the gaps in the input array using graph with only
        forward direction (i.e. time series forecasting)

        :param input_data: data with gaps to filling in the gaps in it
        :return: array without gaps
        """
        output_data = np.array(input_data)

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for gap in new_gap_list:
            # The entire time series is used for training until the gap
            first_gap_element_id = gap[0]
            timeseries_train_part = output_data[:first_gap_element_id]

            # Make forecast in the gap
            predicted = self.__forecast_in_gap(self.pipeline,
                                               timeseries_train_part,
                                               output_data, gap)

            # Replace gaps in an array with prediction values
            output_data[gap] = predicted
        return output_data

    def _forward(self, output_data, batch_index, new_gap_list):
        """
        The time series method makes a forward forecast based on the part
        of the time series that is located to the left of the gap.

        :param output_data: one-dimensional array of a time series
        :param batch_index: index of the interval (batch) with a gap
        :param new_gap_list: array with nested lists of gap indexes

        :return weights_list: numpy array with prediction weights for averaging
        :return predicted: numpy array with prediction values in the gap
        """

        gap = new_gap_list[batch_index]
        first_gap_element_id = gap[0]
        timeseries_train_part = output_data[:first_gap_element_id]

        # Adaptive prediction interval length
        len_gap = len(gap)
        predicted = self.__forecast_in_gap(self.pipeline,
                                           timeseries_train_part,
                                           output_data, gap)
        weights_list = np.arange(len_gap, 0, -1)
        return weights_list, predicted

    def _inverse(self, output_data, batch_index, new_gap_list):
        """
        The time series method makes an inverse forecast based on the part
        of the time series that is located to the right of the gap.

        :param output_data: one-dimensional array of a time series
        :param batch_index: index of the interval (batch) with a gap
        :param new_gap_list: array with nested lists of gap indexes

        :return weights_list: numpy array with prediction weights for averaging
        :return predicted_values: numpy array with prediction values in the gap
        """

        gap = new_gap_list[batch_index]
        # Adaptive prediction interval length
        len_gap = len(gap)

        # If the interval with a gap is the last one in the array
        first_gap_element_id = gap[0]
        latest_gap_element_id = gap[-1]
        if batch_index == len(new_gap_list) - 1:
            timeseries_train_part = output_data[(latest_gap_element_id + 1):]

            if len(timeseries_train_part) == 0:
                # Take last observed value as predicted
                last_known_value = output_data[first_gap_element_id - 1]
                return [last_known_value] * len_gap
        else:
            next_gap = new_gap_list[batch_index + 1]
            timeseries_train_part = output_data[(latest_gap_element_id + 1): next_gap[0]]

            # Take part with known values to the left from the gap
            extended_part = output_data[(first_gap_element_id - 1): next_gap[0]]

            if first_gap_element_id == 0:
                # Gap in the first part of time series - take first observed value
                first_known_value = timeseries_train_part[0]
                return [first_known_value] * len_gap
        timeseries_train_part = np.flip(timeseries_train_part)

        train_ts_len = len(timeseries_train_part) - len_gap
        if train_ts_len < self.min_train_ts_length:
            interpolated_part = self.linear_interpolation(extended_part)
            # Clip pre-history
            interpolated_part = interpolated_part[1:]
            # Clip parts after gap interval
            predicted = interpolated_part[:len_gap]
        else:
            predicted = self.__pipeline_fit_predict(self.pipeline,
                                                    timeseries_train_part,
                                                    len_gap)

            predicted = np.flip(predicted)
        weights_list = np.arange(1, (len_gap + 1), 1)
        return weights_list, predicted

    def __pipeline_fit_predict(self, pipeline, timeseries_train: np.array, len_gap: int):
        """
        The method makes a prediction as a sequence of elements based on a
        training sample. There are two main parts: fit model and predict.

        :param pipeline: pipeline for forecasting
        :param timeseries_train: part of the time series for training the model
        :param len_gap: number of elements in the gap
        :return: array without gaps
        """
        pipeline_for_forecast = deepcopy(pipeline)

        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=len_gap))

        input_data = InputData(idx=np.arange(0, len(timeseries_train)),
                               features=timeseries_train,
                               target=timeseries_train,
                               task=task,
                               data_type=DataTypesEnum.ts)

        # Making predictions for the missing part in the time series
        pipeline_for_forecast.fit_from_scratch(input_data)

        # "Test data" for making prediction for a specific length
        start_forecast = len(timeseries_train)
        end_forecast = start_forecast + len_gap
        idx_test = np.arange(start_forecast, end_forecast)
        test_data = InputData(idx=idx_test,
                              features=timeseries_train,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

        predicted_values = pipeline_for_forecast.predict(test_data)
        predicted_values = np.ravel(np.array(predicted_values.predict))
        return predicted_values

    def __forecast_in_gap(self, pipeline, timeseries_train_part, output_data, gap):
        """ Make forecast for desired part of time series with gap

        :param pipeline: pipeline for forecasting
        :param timeseries_train_part: part of time series without gaps to fit pipeline
        :param output_data: array with gaps (some og them may be filled previously)
        :param gap: indices of continuous batch (gap)
        """

        train_ts_len = len(timeseries_train_part) - len(gap)
        if train_ts_len < self.min_train_ts_length:
            # Take part with gap [..., gap, gap, known_value]
            gap_part = output_data[:gap[-1] + 2]

            # Use linear interpolation
            interpolated_part = self.linear_interpolation(gap_part)
            predicted = interpolated_part[:-1]
        else:
            # Pipeline for the task of filling in gaps
            predicted = self.__pipeline_fit_predict(pipeline,
                                                    timeseries_train_part,
                                                    len(gap))

        return predicted
