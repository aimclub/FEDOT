import math
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from fedot.industrial.core.architecture.settings.computational import backend_methods as np


class TSTransformer:
    """
    Class for transformation single time series based on anomaly dictionary.

    Attributes:
        selected_non_anomaly_intervals: list of non-anomaly intervals which were selected for splitting.

    Example:
        Create time series and anomaly dictionary::
            ts = np.random.rand(800)

        Or for multivariate time series::
            ts = [ts1, ts2, ts3]

        Then create anomaly dictionary::
            anomaly_d_uni = {'anomaly1': [[40, 50], [60, 80], [200, 220], [410, 420], [513, 524], [641, 645]],
                     'anomaly2': [[130, 170], [300, 320], [400, 410], [589, 620], [715, 720]],
                     'anomaly3': [[500, 530], [710, 740]],
                     'anomaly4': [[77, 90], [98, 112], [145, 158], [290, 322]]}

        Split time series into train and test parts::
            from fedot.industrial.core.operation.transformation.splitter import TSSplitter
            splitter = TSSplitter()
            train, test = splitter.transform_for_fit(series=ts, anomaly_dict=anomaly_d_uni, plot=False, binarize=True)

    """

    def __init__(self):
        self.selected_non_anomaly_intervals = []
        self.freq_length = None

    def __check_multivariate(self, time_series: np.ndarray):
        return isinstance(time_series, list) or (
            len(time_series.shape) > 1 and time_series.shape[1] > 1)

    def transform(self, series: np.array):
        """ Split test data on subsequences"""

        return self._transform_test(series)

    def transform_for_fit(
            self,
            series: np.array,
            anomaly_dict: Dict,
            plot: bool = False,
            binarize: bool = False) -> tuple:
        """
        Method for splitting time series into train and test parts based on most frequent anomaly length.

        Args:
            anomaly_dict: dictionary with anomaly labels as keys and anomaly intervals as values.
            series: time series to split.
            plot: if True, plot time series with anomaly intervals.
            binarize: if True, target will be binarized. Recommended for classification task if classes are imbalanced.

        Returns:
            tuple with train and test parts of time series ready for classification task with FedotIndustrial.

        """
        classes = list(anomaly_dict.keys())
        intervals = list(anomaly_dict.values())
        self.freq_length = self._get_frequent_anomaly_length(intervals)
        transformed_intervals = self._transform_intervals(series, intervals)

        features, target = self.get_features_and_target(
            series=series, classes=classes, transformed_intervals=transformed_intervals, binarize=binarize)

        if plot and not self.__check_multivariate(series):
            self.plot_classes_and_intervals(
                series=series,
                classes=classes,
                intervals=intervals,
                transformed_intervals=transformed_intervals)

        return features, target

    def get_features_and_target(
            self,
            series,
            classes,
            transformed_intervals,
            binarize) -> tuple:
        target, features = self._split_by_intervals(
            series, classes, transformed_intervals)
        non_anomaly_inters = self._get_non_anomaly_intervals(
            series, transformed_intervals)
        target, features = self.balance_with_non_anomaly(
            series, target, features, non_anomaly_inters)
        if binarize:
            target = self._binarize_target(target)
        return np.array(features), np.array(target)

    def _get_anomaly_intervals(
            self, anomaly_dict: Dict) -> Tuple[List[str], List[list]]:
        labels = list(anomaly_dict.keys())
        label_intervals = []
        for anomaly_label in labels:
            label_intervals.append(anomaly_dict[anomaly_label])
        return labels, label_intervals

    def _get_frequent_anomaly_length(self, intervals: List[list]):
        flat_intervals = []
        for sublist in intervals:
            for element in sublist:
                flat_intervals.append(element)

        lengths = list(map(lambda x: x[1] - x[0], flat_intervals))
        return max(set(lengths), key=lengths.count)

    def _transform_intervals(self, series, intervals):
        # ts = self.time_series if not self.multivariate else self.time_series.T[0]
        new_intervals = []
        for class_inter in intervals:
            new_class_intervals = []
            for i in class_inter:
                current_len = i[1] - i[0]
                abs_diff = abs(current_len - self.freq_length)
                left_add = math.ceil(abs_diff / 2)
                right_add = math.floor(abs_diff / 2)
                # Calculate new borders

                # If current anomaly interval is less than frequent length,
                # we expand current interval to the size of frequent
                if current_len < self.freq_length:
                    left = i[0] - left_add
                    right = i[1] + right_add
                    # If left border is negative, shift right border to the
                    # right
                    if left < 0:
                        right += abs(left)
                        left = 0
                    # If right border is greater than time series length, shift
                    # left border to the left
                    if right > len(series):
                        left -= abs(right - len(series))
                        right = len(series)

                # If current anomaly interval is greater than frequent length,
                # we shrink current interval to the size of frequent
                elif current_len > self.freq_length:
                    for l in range(i[0], i[1], self.freq_length):
                        new_class_intervals.append([l, l + self.freq_length])
                else:
                    left = i[0]
                    right = i[1]

                    new_class_intervals.append([left, right])
            new_intervals.append(new_class_intervals)
        return new_intervals

    def _split_by_intervals(self,
                            series: np.array,
                            classes: list,
                            transformed_intervals: list) -> Tuple[List[str],
                                                                  List[list]]:
        all_labels, all_ts = [], []

        for i, label in enumerate(classes):
            for inter in transformed_intervals[i]:
                all_labels.append(label)
                if self.__check_multivariate(series):
                    all_ts.append(series[inter[0]:inter[1], :])
                else:
                    all_ts.append(np.ravel(series[inter[0]:inter[1]]))
        return all_labels, all_ts

    def plot_classes_and_intervals(
            self,
            series,
            classes,
            intervals,
            transformed_intervals):
        fig, axes = plt.subplots(3, 1, figsize=(17, 7))
        fig.tight_layout()
        for ax in axes:
            ax.plot(series, color='black', linewidth=1, alpha=1)

        axes[0].set_title('Initial intervals')
        axes[1].set_title('Transformed intervals')
        axes[2].set_title('Non-anomaly samples')

        for i, label in enumerate(classes):
            for interval_ in transformed_intervals[i]:
                axes[1].axvspan(interval_[0], interval_[1],
                                alpha=0.3, color='blue')
                axes[1].text(interval_[0], 0.5, label,
                             fontsize=12, rotation=90)
            for interval in intervals[i]:
                axes[0].axvspan(interval[0], interval[1],
                                alpha=0.3, color='red')
                axes[0].text(interval[0], 0.5, label, fontsize=12, rotation=90)

        if self.selected_non_anomaly_intervals is not None:
            for interval in self.selected_non_anomaly_intervals:
                axes[2].axvspan(interval[0], interval[1],
                                alpha=0.3, color='green')
                axes[2].text(interval[0], 0.5, 'no_anomaly',
                             fontsize=12, rotation=90)
        plt.show()

    def _binarize_target(self, target):
        new_target = []
        for label in target:
            if label == 'no_anomaly':
                new_target.append(0)
            else:
                new_target.append(1)
        return new_target

    def balance_with_non_anomaly(
            self,
            series,
            target,
            features,
            non_anomaly_intervals):
        number_of_anomalies = len(target)
        anomaly_len = len(features[0])
        non_anomaly_ts_list = []
        ts = series.copy()
        counter = 0
        taken_slots = pd.Series([0 for _ in range(len(ts))])

        while len(
                non_anomaly_ts_list) != number_of_anomalies and counter != number_of_anomalies * 100:
            seed = np.random.randint(1000)
            random.seed(seed)
            random_inter = random.choice(non_anomaly_intervals)
            cropped_ts_len = random_inter[1] - random_inter[0]
            counter += 1
            # Exclude intervals that are too short
            if cropped_ts_len < anomaly_len:
                continue
            random_start_index = random.randint(
                random_inter[0], random_inter[0] + cropped_ts_len - anomaly_len)
            stop_index = random_start_index + anomaly_len

            # Check if this interval overlaps with another interval
            if taken_slots[random_start_index:stop_index].mean() > 0.1:
                continue
            else:
                taken_slots[random_start_index:stop_index] = 1

            if self.__check_multivariate(series):
                non_anomaly_ts = ts[random_start_index:stop_index, :]
            else:
                non_anomaly_ts = np.ravel(ts[random_start_index:stop_index])

            non_anomaly_ts_list.append(non_anomaly_ts)

            self.selected_non_anomaly_intervals.append(
                [random_start_index, stop_index])

        if len(non_anomaly_ts_list) == 0:
            raise Exception('No non-anomaly intervals found')

        target.extend(['no_anomaly'] * len(non_anomaly_ts_list))
        features.extend(non_anomaly_ts_list)

        return target, features

    def _get_non_anomaly_intervals(self, series, anom_intervals: List[list]):
        flat_intervals_list = []
        for sublist in anom_intervals:
            for element in sublist:
                flat_intervals_list.append(element)

        if self.__check_multivariate(series):
            series = pd.Series(series[:, 0]).copy()
        else:
            series = pd.Series(np.ravel(series)).copy()

        for single_interval in flat_intervals_list:
            series[single_interval[0]:single_interval[1]] = np.nan

        non_nan_intervals = []
        for k, g in series.groupby(
                (series.notnull() != series.shift().notnull()).cumsum()):
            if g.notnull().any():
                non_nan_intervals.append((g.index[0], g.index[-1]))

        return non_nan_intervals

    def _transform_test(self, series: np.array):
        transformed_data = []
        for i in range(0, series.shape[0], self.freq_length):
            if len(series.shape) == 1:
                series_part = series[i:i + self.freq_length]
                if len(series_part) != self.freq_length:
                    series_part = series[-self.freq_length:]
            else:
                series_part = series[i:i + self.freq_length, :].T
                if series_part.shape[1] != self.freq_length:
                    series_part = series[-self.freq_length:, :].T
            transformed_data.append(series_part)
        transformed_data = np.stack(transformed_data)
        return transformed_data
