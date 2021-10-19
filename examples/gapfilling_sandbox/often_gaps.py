import matplotlib.pyplot as plt
import numpy as np

from examples.time_series.ts_gapfilling_example import generate_gaps_in_ts
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.utilities.synth_dataset_generator import generate_synthetic_data
from fedot.utilities.ts_gapfilling import ModelGapFiller, SimpleGapFiller

if __name__ == '__main__':
    # Generate time series with unknown values at the end and at the start of time series
    gap_arr = np.array([-100.0, -100.0, 2, 3, 5, 6, 2, 3, 1, 4, 5, 4, 5, -100.0, 8, 7, -100.0, 9, 15, 10, 11, -100.0, -100.0, 50, -100.0, -100.0])

    # Much more simple case
    gap_arr = np.array([-100.0, -100.0, 2, 3, 5, 6, 2, 3, 1, 4, 5, 4, 5, 2, 8, 7, -100.0, 9, 15, 10, 11, 1, 5, 50, 2, 1])

    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': 100}
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    ridge_pipeline = Pipeline(node_ridge)
    ridge_gapfiller = ModelGapFiller(gap_value=-100.0,
                                     pipeline=ridge_pipeline)

    without_gap = ridge_gapfiller.forward_inverse_filling(gap_arr)

    plt.plot(gap_arr)
    plt.plot(without_gap)
    plt.show()
