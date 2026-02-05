from typing import Union

import numpy as np
from fedot.core.data.data import InputData


def _apply_park_transform(sample):
    i_1_ch = 1
    i_2_ch = 2
    i_3_ch = 3
    v_1_ch = 4
    v_2_ch = 5
    v_3_ch = 6
    i_alpha = (2 * sample[:i_1_ch, :] - sample[i_1_ch:i_2_ch, :] - sample[i_2_ch:i_3_ch, :]) / 3
    i_beta = (sample[i_1_ch:i_2_ch, :] - sample[i_2_ch:i_3_ch, :]) / np.sqrt(3)
    v_alpha = (2 * sample[i_3_ch:v_1_ch, :] - sample[v_1_ch:v_2_ch, :] - sample[v_2_ch:v_3_ch, :]) / 3
    v_beta = (sample[v_1_ch:v_2_ch, :] - sample[v_2_ch:v_3_ch, :]) / np.sqrt(3)

    # Calculate the instantaneous amplitude and phase of the current and voltage
    instantaneous_i_amplitude = np.sqrt(i_alpha ** 2 + i_beta ** 2)
    instantaneous_i_phase = np.arctan2(i_beta, i_alpha)
    instantaneous_v_amplitude = np.sqrt(v_alpha ** 2 + v_beta ** 2)
    instantaneous_v_phase = np.arctan2(v_beta, v_alpha)
    return np.concatenate([i_alpha, i_beta, v_alpha, v_beta, instantaneous_i_amplitude,
                           instantaneous_i_phase, instantaneous_v_amplitude, instantaneous_v_phase])


def park_transform(input_data: Union[InputData, np.ndarray]) -> np.ndarray:
    """
    Applies the Park transform to a given DataFrame.

    The Park transform is a way to transform 3-phase electrical data into a 2-phase signal, which adds more information.

    Args:
        data (pd.DataFrame): A DataFrame containing the 3-phase electrical data.

    Returns:
        pd.DataFrame: The DataFrame with the added 2-phase electrical data.
    """
    # Calculate the alpha and beta components of the current and voltage
    features = input_data.features if isinstance(input_data, InputData) else input_data
    feature_matrix = list(map(lambda x: _apply_park_transform(x), features))
    return np.stack(feature_matrix)
