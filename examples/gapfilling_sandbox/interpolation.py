import matplotlib.pyplot as plt
import numpy as np

from examples.time_series.ts_gapfilling_example import generate_gaps_in_ts
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.utilities.synth_dataset_generator import generate_synthetic_data
from fedot.utilities.ts_gapfilling import ModelGapFiller, SimpleGapFiller


if __name__ == '__main__':
    # Generate time series with unknown values at the end and at the start of time series
    gap_arr = np.array([-100.0, -100.0, 2, 3, 4, 5, 6, 7, -100.0, 9, 10, 11, -100.0])

    simple_gapfill = SimpleGapFiller(gap_value=-100.0)
    without_gap_linear = simple_gapfill.linear_interpolation(gap_arr)
    without_gap_batch_poly = simple_gapfill.batch_poly_approximation(gap_arr)
    without_gap_local_poly = simple_gapfill.local_poly_approximation(gap_arr, 4, 2)

    plt.plot(gap_arr)
    plt.plot(without_gap_linear)
    plt.plot(without_gap_local_poly)
    plt.show()
