import numpy as np
from sklearn.metrics import mean_squared_error

from examples.simple.time_series_forecasting.gapfilling import get_array_with_gaps
from fedot.utilities.ts_gapfilling import ModelGapFiller, SimpleGapFiller
from test.unit.tasks.test_forecasting import get_simple_ts_pipeline


def test_get_array_with_gaps():
    # Checking whether omissions with the correct values are generated in
    # the right places
    for gap_dict, gap_value in zip([{500: 150}, {700: 10}], [-10.0, -100.0]):
        arr_with_gaps, _ = get_array_with_gaps(gap_dict, gap_value)

        start_gaps = list(gap_dict.keys())
        index_to_check = start_gaps[0]

        assert arr_with_gaps[index_to_check] == gap_value


def test_gapfilling_inverse_ridge_correct():
    """ Testing bidirectional forecast method on the row where gaps are
    frequent, placed at the beginning and at the end of the time series
    """
    arr_with_gaps = np.array([-100, -100, 2, 3, 5, 6, 5, 4, 5, -100, 8, 7, -100,
                              9, 15, 10, 11, -100, -100, 50, -100, -100])

    ridge_pipeline = get_simple_ts_pipeline(model_root='ridge')
    gapfiller = ModelGapFiller(gap_value=-100, pipeline=ridge_pipeline)
    without_gap = gapfiller.forward_inverse_filling(arr_with_gaps)

    assert len(np.ravel(np.argwhere(without_gap == -100))) == 0


def test_gapfilling_forward_ridge_correct():
    arr_with_gaps, real_values = get_array_with_gaps()

    # Find all gap indices in the array
    id_gaps = np.ravel(np.argwhere(arr_with_gaps == -100.0))

    ridge_pipeline = get_simple_ts_pipeline(model_root='ridge')
    gapfiller = ModelGapFiller(gap_value=-100.0, pipeline=ridge_pipeline)
    without_gap = gapfiller.forward_filling(arr_with_gaps)

    # Get only values in the gaps
    predicted_values = without_gap[id_gaps]
    true_values = real_values[id_gaps]

    rmse_test = mean_squared_error(true_values, predicted_values, squared=False)

    assert rmse_test < 1.0


def test_linear_interpolation_fill_start_end():
    """ Linear interpolation can fill in the gaps correctly even gaps are placed
    at the beginning or at the end of time series
    """
    gap_arr = np.array([-100, -100, 2, 3, 4, 5, 6, 7, -100, 9, 10, 11, -100])
    simple_gapfill = SimpleGapFiller(gap_value=-100)
    without_gap_linear = simple_gapfill.linear_interpolation(gap_arr)

    assert len(np.ravel(np.argwhere(without_gap_linear == -100))) == 0


def test_gapfilling_ts_with_no_gaps():
    """ If array doesn't contain gaps, algorithm should return source array """
    no_gap_arr = np.array([1, 1, 2, 3, 4, -50, 6, 7, 8])

    simple_gapfill = SimpleGapFiller(gap_value=-100)
    without_gap = simple_gapfill.local_poly_approximation(no_gap_arr)

    assert tuple(without_gap) == tuple(no_gap_arr)
