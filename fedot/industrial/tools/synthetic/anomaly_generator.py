from typing import Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.tools.synthetic.anomalies import AddNoise, DecreaseDispersion, Dip, IncreaseDispersion, Peak, \
    ShiftTrendDOWN, \
    ShiftTrendUP
from fedot.industrial.tools.synthetic.ts_generator import TimeSeriesGenerator


class AnomalyGenerator:
    """
    AnomalyGenerator class is used to generate anomalies in time series data. It takes time series data as input and
    returns time series data with anomalies. Anomalies are generated based on anomaly_config parameter. It is a dict
    with anomaly class names as keys and anomaly parameters as values. Anomaly class names must be the same as anomaly
    class names in anomalies.py.

    Attributes:
        types: dict. Dict with anomaly class names as keys and anomaly parameters as values.
        anomaly_config: dict, default=BASE_CONFIG. Dict with anomaly class names as keys and anomaly parameters as
            values.
        taken_slots: np.ndarray, default=None. Array of 0 and 1. 1 means that this time slot is already taken by
            another anomaly.
        overlap: float, default=0.1. Argument of `generate` method. Defines the maximum overlap between anomalies.

    Example:
        First, we need to create an instance of AnomalyGenerator class with config as its argument where every anomaly
        type hyperparameters are defined::
            config = {'add_noise': {'level': 80,
                                    'number': 6,
                                    'noise_type': 'gaussian',
                                    'min_anomaly_length': 10,
                                    'max_anomaly_length': 20}}
            generator = AnomalyGenerator(config=config)

        Then we can generate anomalies in time series data using method `generate` which arguments are
        `time_series_data` (`np.array` of config for synthetic ts_data), `plot` and acceptable `overlap`::
            initial_ts, modified_ts, intervals = generator.generate(time_series_data=data,
                                                                    plot=True,
                                                                    overlap=0.1)

        This method returns initial time series data, modified time series data and dict with anomaly intervals.

    """

    def __init__(self, **params):
        self.types = {
            'decrease_dispersion': DecreaseDispersion,
            'increase_dispersion': IncreaseDispersion,
            'shift_trend_up': ShiftTrendUP,
            'shift_trend_down': ShiftTrendDOWN,
            'add_noise': AddNoise,
            'dip': Dip,
            'peak': Peak}

        self.anomaly_config = params.get(
            'config', ValueError('config must be defined'))
        self.taken_slots = None
        self.overlap = None

    def select_interval(self, max_length: int, min_length: int) -> tuple:
        ts_length = self.taken_slots.size
        start_idx = np.random.randint(max_length, ts_length - max_length)
        end_idx = start_idx + np.random.randint(min_length, max_length + 1)

        if self.taken_slots[start_idx:end_idx].mean() > self.overlap:
            return self.select_interval(max_length, min_length)
        else:
            self.taken_slots[start_idx:end_idx] = 1
            return start_idx, end_idx

    def generate(self,
                 time_series_data: Union[np.ndarray,
                                         dict],
                 plot: bool = False,
                 overlap: float = 0.1):
        """
        Generate anomalies in time series data.

        Args:
            time_series_data: either np.ndarray or dict with config for synthetic ts_data.
            plot: if True, plot initial and modified time series data with rectangle spans of anomalies.
            overlap: float, ``default=0.1``. Defines the maximum overlap between anomalies.

        Returns:
            returns initial time series data, modified time series data and dict with anomaly intervals.

        """
        if isinstance(time_series_data, dict):
            ts_generator = TimeSeriesGenerator(time_series_data)
            t_series = ts_generator.get_ts()

        elif isinstance(time_series_data, np.ndarray):
            t_series = time_series_data
        else:
            raise ValueError('time_series_data must be np.ndarray or dict')

        initial_ts = t_series.copy()
        anomaly_intervals_dict = {}

        self.taken_slots = pd.Series([0 for _ in t_series])
        self.overlap = overlap

        for anomaly_cls in self.anomaly_config.keys():
            n = self.anomaly_config[anomaly_cls]['number']
            anomaly_obj = self.types[anomaly_cls]
            params = self.anomaly_config[anomaly_cls]
            max_length = params.get('max_anomaly_length', ValueError(
                f'max_anomaly_length must be defined for {anomaly_cls} type'))
            min_length = params.get('min_anomaly_length', ValueError(
                f'min_anomaly_length must be defined for {anomaly_cls} type'))

            for i in range(n):
                start_idx, end_idx = self.select_interval(
                    max_length, min_length)
                t_series = anomaly_obj(params).get(
                    ts=t_series, interval=(start_idx, end_idx))

                if anomaly_cls in anomaly_intervals_dict:
                    anomaly_intervals_dict[anomaly_cls].append(
                        [start_idx, end_idx])
                else:
                    anomaly_intervals_dict[anomaly_cls] = [
                        [start_idx, end_idx]]

        if plot:
            self.plot_anomalies(initial_ts=initial_ts, modified_ts=t_series,
                                anomaly_intervals_dict=anomaly_intervals_dict)

        return initial_ts, t_series, anomaly_intervals_dict

    def plot_anomalies(self, initial_ts, modified_ts, anomaly_intervals_dict):
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(modified_ts, label='Modified Time Series')
        ax.plot(initial_ts, label='Initial Time Series')
        ax.set_title('Time Series with Anomalies')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

        cmap = self.generate_colors(len(anomaly_intervals_dict.keys()))
        color_dict = {cls: color for cls, color in zip(
            anomaly_intervals_dict.keys(), cmap)}

        legend_patches = [
            patches.Patch(
                color=color_dict[cls],
                label=cls) for cls in anomaly_intervals_dict.keys()]

        for anomaly_class, intervals in anomaly_intervals_dict.items():
            for interval in intervals:
                start_idx, end_idx = interval
                ax.axvspan(start_idx, end_idx, alpha=0.3,
                           color=color_dict[anomaly_class])

        # Put a legend to the right of the current axis
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(
            1, 0.5), handles=set(legend_patches))
        plt.show()

    def generate_colors(self, num_colors: int) -> list:
        colormap = plt.cm.get_cmap('tab10')
        colors = [colormap(i) for i in range(num_colors)]
        return colors


if __name__ == '__main__':

    # config = {'decrease_dispersion': {'level': 70,
    #                                   'number': 2,
    #                                   'min_anomaly_length': 10,
    #                                   'max_anomaly_length': 15},
    #           'dip': {'level': 20,
    #                   'number': 2,
    #                   'min_anomaly_length': 10,
    #                   'max_anomaly_length': 20},
    #
    #           'peak': {'level': 2,
    #                    'number': 2,
    #                    'min_anomaly_length': 5,
    #                    'max_anomaly_length': 10},
    #           'increase_dispersion': {'level': 70,
    #                                   'number': 2,
    #                                   'min_anomaly_length': 30,
    #                                   'max_anomaly_length': 40},
    #           'shift_trend_up': {'level': 10,
    #                              'number': 2,
    #                              'min_anomaly_length': 10,
    #                              'max_anomaly_length': 20},
    #           'shift_trend_down': {'level': 10,
    #                                'number': 2,
    #                                'min_anomaly_length': 10,
    #                                'max_anomaly_length': 20},
    #           'add_noise': {'level': 80,
    #                         'number': 2,
    #                         'noise_type': 'uniform',
    #                         'min_anomaly_length': 50,
    #                         'max_anomaly_length': 60}
    #           }
    #
    # generator = AnomalyGenerator(config=config)
    #
    # ts_conf = {'ts_type': 'sin',
    #            'ts_length': 2000}
    #
    # init_ts, mot_ts, inters = generator.generate(time_series_data=ts_conf,
    #                                              plot=True,
    #                                              overlap=0.1)

    synth_ts = {'ts_type': 'sin',
                'length': 1000,
                'amplitude': 10,
                'period': 500}

    anomaly_config = {'dip': {'level': 20,
                              'number': 5,
                              'min_anomaly_length': 10,
                              'max_anomaly_length': 20},
                      'peak': {'level': 2,
                               'number': 5,
                               'min_anomaly_length': 5,
                               'max_anomaly_length': 10},
                      'decrease_dispersion': {'level': 70,
                                              'number': 2,
                                              'min_anomaly_length': 10,
                                              'max_anomaly_length': 15},
                      # 'increase_dispersion': {'level': 50,
                      #                         'number': 2,
                      #                         'min_anomaly_length': 10,
                      #                         'max_anomaly_length': 15},
                      # 'shift_trend_up': {'level': 10,
                      #                    'number': 2,
                      #                    'min_anomaly_length': 10,
                      #                    'max_anomaly_length': 20},
                      # 'add_noise': {'level': 80,
                      #               'number': 2,
                      #               'noise_type': 'uniform',
                      #               'min_anomaly_length': 10,
                      #               'max_anomaly_length': 20}
                      }

    generator = AnomalyGenerator(config=anomaly_config)

    init_synth_ts, mod_synth_ts, synth_inters = generator.generate(
        time_series_data=synth_ts, plot=True, overlap=0.1)
    _ = 1
