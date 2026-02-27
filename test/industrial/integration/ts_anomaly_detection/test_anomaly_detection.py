from typing import Dict

from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.architecture.settings.computational import backend_methods as np
import pytest
from fedot.api.main import Fedot

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.repository.config_repository import DEFAULT_CLF_API_CONFIG


def convert_anomalies_dict_to_points(
        series: np.array,
        anomaly_dict: Dict) -> np.array:
    points = np.array(['no_anomaly' for _ in range(len(series))], dtype=object)
    for anomaly_class in anomaly_dict:
        for interval in anomaly_dict[anomaly_class]:
            points[interval[0]:interval[1]] = anomaly_class
    return points


def split_series(series, anomaly_dict, test_part: int = 200):
    time_series_train = series[:-test_part]
    time_series_test = series[-test_part:]

    anomaly_intervals_train = {}
    anomaly_intervals_test = {}
    for anomaly_class in anomaly_dict:
        single_class_anomalies_train = []
        single_class_anomalies_test = []
        for interval in anomaly_dict[anomaly_class]:
            if interval[1] > len(time_series_train):
                single_class_anomalies_test.append(
                    [interval[0] - len(time_series_train), interval[1] - len(time_series_train)])
            else:
                single_class_anomalies_train.append(interval)
        anomaly_intervals_train[anomaly_class] = single_class_anomalies_train
        anomaly_intervals_test[anomaly_class] = single_class_anomalies_test
    return time_series_train, anomaly_intervals_train, time_series_test, anomaly_intervals_test


def generate_time_series(ts_length: int = 500,
                         dimension: int = 1,
                         num_anomaly_classes: int = 4,
                         num_of_anomalies: int = 20,
                         min_anomaly_length: int = 5,
                         max_anomaly_length: int = 15):
    np.random.seed(42)

    if dimension == 1:
        time_series = np.random.normal(0, 1, ts_length)
    else:
        time_series = np.vstack([np.random.normal(0, 1, ts_length)
                                 for _ in range(dimension)]).swapaxes(1, 0)
    anomaly_classes = [f'anomaly{i + 1}' for i in range(num_anomaly_classes)]

    anomaly_intervals = {}

    for i in range(num_of_anomalies):
        anomaly_class = np.random.choice(anomaly_classes)

        start_idx = np.random.randint(max_anomaly_length,
                                      ts_length - max_anomaly_length)

        end_idx = start_idx + np.random.randint(min_anomaly_length,
                                                max_anomaly_length + 1)

        anomaly = np.random.normal(
            int(anomaly_class[-1]), 1, end_idx - start_idx)

        if dimension == 1:
            time_series[start_idx:end_idx] += anomaly
        else:
            for j in range(time_series.shape[1]):
                time_series[start_idx:end_idx, j] += anomaly

        if anomaly_class in anomaly_intervals:
            anomaly_intervals[anomaly_class].append([start_idx, end_idx])
        else:
            anomaly_intervals[anomaly_class] = [[start_idx, end_idx]]

    return time_series, anomaly_intervals


@pytest.mark.skip('Anomaly detection skipped due to invalid data generation')
@pytest.mark.parametrize('dimension', [1, 3])
def test_anomaly_detection(dimension):
    np.random.seed(42)
    time_series, anomaly_intervals = generate_time_series(ts_length=1000,
                                                          dimension=dimension,
                                                          num_anomaly_classes=2,
                                                          num_of_anomalies=50)

    series_train, anomaly_train, series_test, anomaly_test = split_series(time_series,
                                                                          anomaly_intervals,
                                                                          test_part=300)
    point_train = convert_anomalies_dict_to_points(series_train, anomaly_train)
    point_test = convert_anomalies_dict_to_points(series_test, anomaly_test)

    config = dict(task='classification',
                  strategy='fedot_preset',
                  timeot=1,
                  n_jobs=-1,
                  logging_level=20)
    api_config = ApiConfigCheck().update_config_with_kwargs(DEFAULT_CLF_API_CONFIG,
                                                            **config)

    industrial = FedotIndustrial(**api_config)

    model = industrial.fit((series_train, point_train))

    # industrial.solver.save('model')

    # prediction before loading
    labels_before = industrial.predict((series_test, point_test))
    probs_before = industrial.predict_proba((series_test, point_test))

    # industrial.solver.load('model')

    # prediction after loading
    # labels_after = industrial.predict(features=series_test)
    # probs_after = industrial.predict_proba(features=series_test)

    metrics = industrial.get_metrics(labels=labels_before,
                                     probs=probs_before,
                                     target=point_test,
                                     metric_names=('f1', 'roc_auc'))

    # shutil.rmtree('model')
    # assert np.all(labels_after == labels_before)

    assert metrics['f1'] > 0.5
    assert metrics['roc_auc'] > 0.5
    assert isinstance(model, Fedot)
    assert isinstance(labels_before, list)
    assert isinstance(probs_before, list)
