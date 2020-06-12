import random
import pytest

import numpy as np
from core.composer.node import PrimaryNode
from core.composer.ts_chain import TsForecastingChain
from utilities.ts_gapfilling import ModelGapFiller
from sklearn.metrics import mean_squared_error
from examples.time_series_gapfilling_example import generate_synthetic_data


def test_gapfilling_inverse_ridge_correct():
    arr_with_gaps, real_values = generate_synthetic_data(length=1000,
                                                         gap_size=40,
                                                         gap_value=-100.0,
                                                         periods=6,
                                                         border=400)

    # Find all gap indices in the array
    id_gaps = np.argwhere(arr_with_gaps == -100.0)

    ridge_chain = TsForecastingChain(PrimaryNode('ridge'))
    gapfiller = ModelGapFiller(gap_value=-100.0, chain=ridge_chain)
    without_gap = gapfiller.forward_inverse_filling(arr_with_gaps, 100)
    predicted_values = without_gap[id_gaps]

    rmse_test = mean_squared_error(real_values, predicted_values, squared=False)

    # The RMSE must be less than the standard deviation of random noise * 1.5
    assert rmse_test < 0.15


def test_gapfilling_forward_ridge_correct():
    arr_with_gaps, real_values = generate_synthetic_data(length=1000,
                                                         gap_size=40,
                                                         gap_value=-100.0,
                                                         periods=6,
                                                         border=400)

    # Find all gap indices in the array
    id_gaps = np.argwhere(arr_with_gaps == -100.0)

    ridge_chain = TsForecastingChain(PrimaryNode('ridge'))
    gapfiller = ModelGapFiller(gap_value=-100.0, chain=ridge_chain)
    without_gap = gapfiller.forward_filling(arr_with_gaps, 100)
    predicted_values = without_gap[id_gaps]

    rmse_test = mean_squared_error(real_values, predicted_values, squared=False)

    # The RMSE must be less than the standard deviation of random noise * 2.0
    assert rmse_test < 0.2
