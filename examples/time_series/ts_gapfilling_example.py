import matplotlib.pyplot as plt
import numpy as np

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.utilities.ts_gapfilling import ModelGapFiller, SimpleGapFiller
from data.data_manager import get_array_with_gaps


def run_gapfilling_example():
    """
    This function runs an example of filling in gaps in synthetic data

    :return arrays_dict: dictionary with 4 keys ('ridge', 'local_poly',
    'batch_poly', 'linear') that can be used to get arrays without gaps
    :return gap_data: an array with gaps
    :return real_data: an array with actual values in gaps
    """

    # Get synthetic time series
    gap_data, real_data = get_array_with_gaps()

    # Filling in gaps using pipeline from FEDOT
    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': 100}
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    ridge_pipeline = Pipeline(node_ridge)
    ridge_gapfiller = ModelGapFiller(gap_value=-100.0,
                                     pipeline=ridge_pipeline)
    without_gap_arr_ridge = \
        ridge_gapfiller.forward_inverse_filling(gap_data)

    # Filling in gaps using simple methods such as polynomial approximation
    simple_gapfill = SimpleGapFiller(gap_value=-100.0)
    without_gap_local_poly = \
        simple_gapfill.local_poly_approximation(gap_data, 4, 150)

    without_gap_batch_poly = \
        simple_gapfill.batch_poly_approximation(gap_data, 4, 150)

    without_gap_linear = \
        simple_gapfill.linear_interpolation(gap_data)

    arrays_dict = {'ridge': without_gap_arr_ridge,
                   'local_poly': without_gap_local_poly,
                   'batch_poly': without_gap_batch_poly,
                   'linear': without_gap_linear}
    return arrays_dict, gap_data, real_data


def plot_results(arrays_dict, gap_data):
    """
    Plot predictions of models

    :param arrays_dict: dictionary with 4 keys ('ridge', 'local_poly',
    'batch_poly', 'linear') that can be used to get arrays without gaps
    :param gap_data: an array with gaps
    """

    gap_ids = np.ravel(np.argwhere(gap_data == -100.0))
    masked_array = np.ma.masked_where(gap_data == -100.0, gap_data)

    plt.plot(arrays_dict.get('local_poly'), c='orange',
             alpha=0.5, label='Local polynomial approximation')
    plt.plot(arrays_dict.get('batch_poly'), c='green',
             alpha=0.5, label='Batch polynomial approximation')
    plt.plot(arrays_dict.get('linear'), c='black',
             alpha=0.5, label='Linear interpolation')
    plt.plot(arrays_dict.get('ridge'), c='red',
             alpha=0.5, label='Inverse ridge')
    plt.plot(masked_array, c='blue', alpha=1.0, label='Actual values')
    plt.xlim(gap_ids[0] - 100, gap_ids[-1] + 100)
    plt.legend(loc='upper center')
    plt.grid()
    plt.show()


# Example of applying the algorithm
if __name__ == '__main__':
    arrays_dict, gap_data, _ = run_gapfilling_example()

    # Make the plots with predictions of different models
    plot_results(arrays_dict, gap_data)
