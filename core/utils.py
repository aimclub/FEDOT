from pathlib import Path
import numpy as np
import pandas as pd

def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent

# Rolling 2D window for ND array
def roll(a,      # ND array
         shape,  # rolling 2D window array
         dx=1,   # horizontal step, abscissa, number of columns
         dy=1):  # vertical step, ordinate, number of rows

    shape = a.shape[:-2] + \
            ((a.shape[-2] - shape[-2]) // dy + 1,) + \
            ((a.shape[-1] - shape[-1]) // dx + 1,) + \
            shape  # sausage-like shape with 2D cross-section
    strides = a.strides[:-2] + \
              (a.strides[-2] * dy,) + \
              (a.strides[-1] * dx,) + \
              a.strides[-2:]

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def ts_to_3d(array, window_len: int):
    """
    Makes 3d dataset of sliding windows with shape (n-window_len+1, window_len, features)
    from (n, features) array.
    array: np.ndarray or pd.DataFrame
    """
    if isinstance(array, pd.DataFrame):
        array = array.to_numpy()
    features = array.shape[1]
    res3d = roll(array, (window_len, features)).reshape(-1, window_len, features)
    return res3d

