from typing import Union
import torch
import math
from fedot.core.data.data import InputData


def park_transform_torch(
    input_data: Union["InputData", torch.Tensor]
) -> torch.Tensor:
    """
    Applies the Park transform to 3-phase electrical signals in a vectorized
    manner.

    The Park transform converts 3-phase current and voltage signals
    (i1, i2, i3, v1, v2, v3)into α-β components and computes instantaneous
    amplitude and phase for both current and voltage. This transformation is
    useful for analyzing electrical signals in rotating reference frames.

    Args:
        input_data: InputData object or torch.Tensor of shape (N, 6, T), where:
                    - N: Number of samples in the batch
                    - 6: Three current phases (i1, i2, i3) and three voltage
                    phases (v1, v2, v3)
                    - T: Time samples

    Returns:
        torch.Tensor: Transformed tensor of shape (N, 8, T) containing:
                      - i_alpha, i_beta: α-β components of current
                      - v_alpha, v_beta: α-β components of voltage
                      - instantaneous_i_amplitude, instantaneous_i_phase:
                      Amplitude and phase of current
                      - instantaneous_v_amplitude, instantaneous_v_phase:
                      Amplitude and phase of voltage
    """
    features = (input_data.features if hasattr(input_data, "features")
                else input_data)

    # features: (N, 6, T)
    i1, i2, i3, v1, v2, v3 = features.unbind(dim=1)

    sqrt3 = math.sqrt(3.0)

    i_alpha = (2.0 * i1 - i2 - i3) / 3.0
    i_beta = (i2 - i3) / sqrt3

    v_alpha = (2.0 * v1 - v2 - v3) / 3.0
    v_beta = (v2 - v3) / sqrt3

    instantaneous_i_amplitude = torch.sqrt(i_alpha**2 + i_beta**2)
    instantaneous_i_phase = torch.atan2(i_beta, i_alpha)

    instantaneous_v_amplitude = torch.sqrt(v_alpha**2 + v_beta**2)
    instantaneous_v_phase = torch.atan2(v_beta, v_alpha)

    return torch.stack(
        [
            i_alpha,
            i_beta,
            v_alpha,
            v_beta,
            instantaneous_i_amplitude,
            instantaneous_i_phase,
            instantaneous_v_amplitude,
            instantaneous_v_phase,
        ],
        dim=1,
    )
