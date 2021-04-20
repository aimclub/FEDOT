import matplotlib.pyplot as plt
import numpy as np

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.utilities.synth_dataset_generator import generate_synthetic_data
from fedot.utilities.ts_gapfilling import ModelGapFiller, SimpleGapFiller


def generate_gaps_in_ts(array_without_gaps, gap_dict, gap_value):
    """
    Function for generating gaps with predefined length in the desired indices
    of an one-dimensional array (time series)

    :param array_without_gaps: an array without gaps
    :param gap_dict: a dictionary with omissions, where the key is the index in
    the time series from which the gap will begin. The key value is the length
    of the gap (elements). -1 in the value means that a skip is generated until
    the end of the array
    :param gap_value: value indicating a gap in the array

    :return: one-dimensional array with omissions
    """

    array_with_gaps = np.copy(array_without_gaps)

    keys = list(gap_dict.keys())
    for key in keys:
        gap_size = gap_dict.get(key)
        if gap_size == -1:
            # Generating a gap to the end of an array
            array_with_gaps[key:] = gap_value
        else:
            array_with_gaps[key:(key + gap_size)] = gap_value

    return array_with_gaps


def get_array_with_gaps(gap_dict=None, gap_value: float = -100.0):
    """
    Function for generating synthetic data and gaps in it with predefined length
    and location

    :param gap_dict: a dictionary with omissions, where the key is the index in
    the time series from which the gap will begin. The key value is the length
    of the gap (elements). -1 in the value means that a skip is generated until
    the end of the array
    :param gap_value: value indicating a gap in the array

    :return array_with_gaps: an array with gaps
    :return real_values: an array with actual values in gaps
    """

    real_values = generate_synthetic_data()

    if gap_dict is None:
        gap_dict = {850: 100,
                    1400: 150}
    array_with_gaps = generate_gaps_in_ts(array_without_gaps=real_values,
                                          gap_dict=gap_dict,
                                          gap_value=gap_value)

    return array_with_gaps, real_values


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

    # Filling in gaps using chain from FEDOT
    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': 100}
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    ridge_chain = Chain(node_ridge)
    ridge_gapfiller = ModelGapFiller(gap_value=-100.0,
                                     chain=ridge_chain)
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
