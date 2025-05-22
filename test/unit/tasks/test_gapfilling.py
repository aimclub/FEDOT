import numpy as np
import pytest
from typing import Callable

from examples.simple.time_series_forecasting.gapfilling import get_array_with_gaps
from fedot.core.composer.metrics import root_mean_squared_error
from fedot.utilities.ts_gapfilling import ModelGapFiller, SimpleGapFiller
from test.unit.tasks.test_forecasting import get_simple_ts_pipeline


def gap_time_series_first_gap_enough_length():
    """
    First element in array is a gap but time series length enough for model
    training
    """
    ts = [-100.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    return np.array(ts, dtype=float)


def gap_time_series_first_gap_not_enough_length():
    """
    First element in array is a gap and time series length does not enough
    for model training
    """
    ts = [-100.0, 1, 2, 3]
    return np.array(ts, dtype=float)


def gap_time_series_last_gap_enough_length():
    """
    Last element in array is a gap but time series length enough for model
    training
    """
    ts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -100.0]
    return np.array(ts, dtype=float)


def gap_time_series_last_gap_not_enough_length():
    """
    First element in array is a gap and time series length does not enough
    for model training
    """
    ts = [0, 1, 2, 3, -100.0]
    return np.array(ts, dtype=float)


def gap_time_series_first_last_elements_enough_length():
    """ First and last elements are gaps """
    ts = [-100.0, -100.0, -100.0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
          16, 17, 18, 19, 20, 21, -100.0, -100.0, -100.0]
    return np.array(ts, dtype=float)


def gap_time_series_first_last_elements_not_enough_length():
    """ First and last elements are gaps """
    ts = [-100.0, -100.0, -100.0, 3, 4, 5, 6, -100.0, -100.0, -100.0]
    return np.array(ts, dtype=float)


def gap_time_series_several_first_last_enough_length():
    """ Some first and last elements are gaps but not the first and last one """
    ts = [0, -100.0, -100.0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
          -100.0, -100.0, 20]
    return np.array(ts, dtype=float)


def gap_time_series_several_first_last_not_enough_length():
    """ Some first and last elements are gaps but not the first and last one """
    ts = [0, -100.0, -100.0, 3, 4, 5, 6, -100.0, -100.0, 9]
    return np.array(ts, dtype=float)


def gap_time_series_large_gaps_ratio():
    """ Case when number of gap elements is big """
    ts = [0, -100.0, -100.0, 3, 4, -100.0, -100.0, -100.0, -100.0, -100.0,
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, -100.0, -100.0, -100.0,
          24, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, 31]
    return np.array(ts, dtype=float)


def gap_time_series_complicated_gaps_configuration():
    """
    Generate complicated case for time series gap filling. Array also contain
    np.nan values
    """
    ts = [-100.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -100.0,
          np.nan, 19, 20, 21, 22, 23, 24, 25, 26, -100.0, 28, 29, 30, 31]
    return np.array(ts, dtype=float)


TIME_SERIES_GAPS_SETUPS = [gap_time_series_first_gap_enough_length,
                           gap_time_series_first_gap_not_enough_length,
                           gap_time_series_last_gap_enough_length,
                           gap_time_series_last_gap_not_enough_length,
                           gap_time_series_first_last_elements_enough_length,
                           gap_time_series_first_last_elements_not_enough_length,
                           gap_time_series_several_first_last_enough_length,
                           gap_time_series_several_first_last_not_enough_length,
                           gap_time_series_large_gaps_ratio,
                           gap_time_series_complicated_gaps_configuration]


def test_get_array_with_gaps():
    """
    Checking whether omissions with the correct values are generated in
    the right places
    """
    for gap_dict, gap_value in zip([{10: 10}, {50: 20}], [-10.0, -100.0]):
        arr_with_gaps, _ = get_array_with_gaps(gap_dict, gap_value)

        start_gaps = list(gap_dict.keys())
        index_to_check = start_gaps[0]

        assert arr_with_gaps[index_to_check] == gap_value


def test_gap_filling_ts_with_no_gaps():
    """ If array doesn't contain gaps, algorithm should return source array """
    no_gap_arr = np.array([1, 1, 2, 3, 4, -50, 6, 7, 8])

    simple_gapfill = SimpleGapFiller(gap_value=-100)
    without_gap = simple_gapfill.local_poly_approximation(no_gap_arr)

    assert tuple(without_gap) == tuple(no_gap_arr)


@pytest.mark.parametrize("get_time_series", TIME_SERIES_GAPS_SETUPS)
def test_gap_filling_forward_correct(get_time_series: Callable):
    """
    Goal: check that assimilation different time series does not leads to errors
    and missing values presence
    """
    arr_with_gaps = get_time_series()

    linear_pipeline = get_simple_ts_pipeline(model_root='linear')
    gapfiller = ModelGapFiller(gap_value=-100, pipeline=linear_pipeline)
    without_gap = gapfiller.forward_filling(arr_with_gaps)

    assert len(np.ravel(np.argwhere(without_gap == -100))) == 0


@pytest.mark.parametrize("get_time_series", TIME_SERIES_GAPS_SETUPS)
def test_gap_filling_forward_inverse_correct(get_time_series: Callable):
    """ Testing bidirectional forecast method on the row where gaps are
    frequent, placed at the beginning and at the end of the time series
    Goal: check that assimilation different time series does not leads to errors
    and missing values presence
    """
    arr_with_gaps = get_time_series()

    linear_pipeline = get_simple_ts_pipeline(model_root='linear')
    gapfiller = ModelGapFiller(gap_value=-100, pipeline=linear_pipeline)
    without_gap = gapfiller.forward_inverse_filling(arr_with_gaps)

    assert len(np.ravel(np.argwhere(without_gap == -100))) == 0


@pytest.mark.parametrize("get_time_series", TIME_SERIES_GAPS_SETUPS)
def test_linear_interpolation_correct(get_time_series: Callable):
    """ Linear interpolation can fill in the gaps correctly even gaps are placed
    at the beginning or at the end of time series (and other cases)
    """
    gap_arr = get_time_series()
    simple_gapfill = SimpleGapFiller(gap_value=-100)
    without_gap_linear = simple_gapfill.linear_interpolation(gap_arr)

    assert len(np.ravel(np.argwhere(without_gap_linear == -100))) == 0


@pytest.mark.parametrize("get_time_series", TIME_SERIES_GAPS_SETUPS)
def test_local_polynomial_interpolation_correct(get_time_series: Callable):
    gap_arr = get_time_series()
    simple_gapfill = SimpleGapFiller(gap_value=-100)
    without_gap_local_poly = simple_gapfill.local_poly_approximation(gap_arr)

    assert len(np.ravel(np.argwhere(without_gap_local_poly == -100))) == 0


@pytest.mark.parametrize("get_time_series", TIME_SERIES_GAPS_SETUPS)
def test_batch_polynomial_interpolation_correct(get_time_series: Callable):
    gap_arr = get_time_series()
    simple_gapfill = SimpleGapFiller(gap_value=-100)
    without_gap_batch_poly = simple_gapfill.batch_poly_approximation(gap_arr)

    assert len(np.ravel(np.argwhere(without_gap_batch_poly == -100))) == 0


def test_gap_filling_forward_ridge_metric_correct():
    """
    Check that gap filling module allows to get appropriate error value into
    the filled parts
    """
    arr_with_gaps, real_values = get_array_with_gaps()

    # Find all gap indices in the array
    id_gaps = np.ravel(np.argwhere(arr_with_gaps == -100.0))

    ridge_pipeline = get_simple_ts_pipeline(model_root='ridge')
    gapfiller = ModelGapFiller(gap_value=-100.0, pipeline=ridge_pipeline)
    without_gap = gapfiller.forward_filling(arr_with_gaps)

    # Get only values in the gaps
    predicted_values = without_gap[id_gaps]
    true_values = real_values[id_gaps]

    rmse_test = root_mean_squared_error(true_values, predicted_values)

    assert rmse_test < 1.0
