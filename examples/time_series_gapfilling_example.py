import random

import matplotlib.pyplot as plt
import numpy as np

from fedot.core.composer.node import PrimaryNode
from fedot.core.composer.ts_chain import TsForecastingChain
from fedot.utilities.ts_gapfilling import SimpleGapFiller, ModelGapFiller


def generate_synthetic_data(length: int = 2500, gap_size: int = 100,
                            gap_value: float = -100.0, periods: int = 6,
                            border: int = 1000):
    """
    The function generates a synthetic one-dimensional array with omissions

    :param length: the length of the array (should be more than 1000)
    :param gap_size: number of elements in the gap
    :param gap_value: value, which identify gap elements in array
    :param periods: the number of periods in the sine wave
    :param border: minimum number of known time series elements before and after
    the gap
    :return synthetic_data: an array with gaps
    :return real_values: an array with actual values in gaps
    """

    sinusoidal_data = np.linspace(-periods * np.pi, periods * np.pi, length)
    sinusoidal_data = np.sin(sinusoidal_data)
    random_noise = np.random.normal(loc=0.0, scale=0.1, size=length)

    # Combining a sine wave and random noise
    synthetic_data = sinusoidal_data + random_noise

    random_value = random.randint(border, length - border)
    real_values = np.array(
        synthetic_data[random_value:(random_value + gap_size)])
    synthetic_data[random_value: (random_value + gap_size)] = gap_value
    return synthetic_data, real_values


def run_gapfilling_example():
    """
    This function runs an example of filling in gaps in synthetic data

    :return arrays_dict: dictionary with 4 keys ('ridge', 'local_poly',
    'batch_poly', 'linear') that can be used to get arrays without gaps
    :return gap_data: an array with gaps
    :return real_data: an array with actual values in gaps
    """

    # Get synthetic time series
    gap_data, real_data = generate_synthetic_data()

    # Filling in gaps using chain from FEDOT
    ridge_chain = TsForecastingChain(PrimaryNode('ridge'))
    ridge_gapfiller = ModelGapFiller(gap_value=-100.0,
                                     chain=ridge_chain)
    without_gap_arr_ridge = \
        ridge_gapfiller.forward_inverse_filling(gap_data, 400)

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


# Example of applying the algorithm
if __name__ == '__main__':
    arrays_dict, gap_data, _ = run_gapfilling_example()

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
    plt.legend()
    plt.grid()
    plt.show()
