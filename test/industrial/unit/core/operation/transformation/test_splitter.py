import warnings

import numpy as np
import pytest
from matplotlib import get_backend, pyplot as plt

from fedot_ind.core.operation.transformation.splitter import TSTransformer


@pytest.fixture
def frequent_splitter():
    return TSTransformer()


@pytest.fixture
def time_series():
    return np.random.rand(320)


@pytest.fixture
def anomaly_dict():
    return {'anomaly1': [[40, 50], [60, 80]],
            'anomaly2': [[130, 170], [300, 320]]}


def test__check_multivariate(frequent_splitter, time_series):
    univariate = frequent_splitter._TSTransformer__check_multivariate(
        time_series)
    multivariate = frequent_splitter._TSTransformer__check_multivariate(
        np.array([time_series, time_series]))
    assert univariate is False
    assert multivariate is True


def test_transform(frequent_splitter, time_series, anomaly_dict):
    frequent_splitter.freq_length = 20
    transformed_data = frequent_splitter._transform_test(time_series)
    assert isinstance(transformed_data, np.ndarray)
    assert transformed_data.shape[1] == frequent_splitter.freq_length


@pytest.mark.parametrize('binarize, plot', ([True, False], [False, False],
                                            [True, True], [False, True]))
def test_transform_for_fit(
        frequent_splitter,
        time_series,
        anomaly_dict,
        binarize,
        plot):
    # switch to non-Gui, preventing plots being displayed
    # suppress UserWarning that agg cannot show plots
    get_backend()
    plt.switch_backend("Agg")
    warnings.filterwarnings("ignore", "Matplotlib is currently using agg")
    features, target = frequent_splitter.transform_for_fit(
        series=time_series, anomaly_dict=anomaly_dict, plot=plot, binarize=binarize)
    assert isinstance(features, np.ndarray)
    assert isinstance(target, np.ndarray)
    if binarize:
        assert np.mean(target) == 0.5
    else:
        assert np.mean(target == 'no_anomaly') == 0.5


@pytest.mark.parametrize('binarize', (True, False))
def test_get_features_and_target(frequent_splitter,
                                 time_series,
                                 anomaly_dict,
                                 binarize):
    classes = list(anomaly_dict.keys())
    intervals = list(anomaly_dict.values())
    frequent_splitter.freq_length = 20
    trans_intervals = frequent_splitter._transform_intervals(
        series=time_series, intervals=intervals)
    features, target = frequent_splitter.get_features_and_target(
        series=time_series, classes=classes, transformed_intervals=trans_intervals, binarize=binarize)

    assert isinstance(features, np.ndarray)
    assert isinstance(target, np.ndarray)
    if binarize:
        assert np.mean(target) == 0.5
    else:
        assert np.mean(target == 'no_anomaly') == 0.5


def test__get_anomaly_intervals(frequent_splitter, anomaly_dict):
    labels, label_intervals = frequent_splitter._get_anomaly_intervals(
        anomaly_dict=anomaly_dict)
    assert isinstance(labels, list)
    assert isinstance(label_intervals, list)


def test__get_frequent_anomaly_length(frequent_splitter, anomaly_dict):
    inters = list(anomaly_dict.values())
    value = frequent_splitter._get_frequent_anomaly_length(inters)

    assert value
    assert isinstance(value, int)


def test__transform_intervals(frequent_splitter, time_series, anomaly_dict):
    inters = list(anomaly_dict.values())
    frequent_splitter.freq_length = 20
    new_intervals = frequent_splitter._transform_intervals(time_series,
                                                           inters)

    assert isinstance(new_intervals, list)
    assert len(new_intervals) == len(anomaly_dict.keys())


def test__split_by_intervals(frequent_splitter, time_series, anomaly_dict):
    classes = list(anomaly_dict.keys())
    intervals = list(anomaly_dict.values())
    frequent_splitter.freq_length = 20
    transformed_intervals = frequent_splitter._transform_intervals(
        series=time_series, intervals=intervals)

    all_labels, all_ts = frequent_splitter._split_by_intervals(
        series=time_series, classes=classes, transformed_intervals=transformed_intervals)

    assert isinstance(all_ts, list)
    assert isinstance(all_ts, list)
    for i in anomaly_dict:
        assert i in all_labels
    for fragment in all_ts:
        assert len(fragment) == frequent_splitter.freq_length


def test_binarize_target(frequent_splitter):
    target = ['anomaly', 'anomaly', 'anomaly']
    new_target = frequent_splitter._binarize_target(target=target)
    assert np.mean(new_target == 'no_anomaly') == 0
    assert np.mean(new_target == 'anomaly') == 0


def test_balance_with_non_anomaly(
        frequent_splitter,
        time_series,
        anomaly_dict):
    classes = list(anomaly_dict.keys())
    intervals = list(anomaly_dict.values())
    frequent_splitter.freq_length = 20
    transformed_intervals = frequent_splitter._transform_intervals(
        series=time_series, intervals=intervals)
    target, features, = frequent_splitter._split_by_intervals(
        time_series, classes, transformed_intervals)
    non_anomaly_inters = frequent_splitter._get_non_anomaly_intervals(
        time_series, transformed_intervals)
    new_target, new_features = frequent_splitter.balance_with_non_anomaly(
        time_series, target, features, non_anomaly_inters)

    assert new_target.count('no_anomaly') / len(new_target) == 0.5
    assert len(new_target) == len(new_features)


def test_get_non_anomaly_intervals(
        frequent_splitter,
        anomaly_dict,
        time_series):
    intervals = list(anomaly_dict.values())
    frequent_splitter.freq_length = 20
    transformed_intervals = frequent_splitter._transform_intervals(
        series=time_series, intervals=intervals)

    non_nan_intervals = frequent_splitter._get_non_anomaly_intervals(
        time_series, transformed_intervals)
    ts_len = len(time_series)
    assert isinstance(non_nan_intervals, list)
    assert non_nan_intervals[0][0] in range(ts_len)
    assert non_nan_intervals[-1][1] in range(ts_len)
