from typing import Union, Optional, List
import os
import pandas as pd
import numpy as np
#import cudf
import cupy as cp


PathType = Union[os.PathLike, str]

PandasType = Union[pd.DataFrame, pd.Series, #cudf.DataFrame, cudf.Series
]

ArrayType = Optional[Union[np.ndarray, cp.ndarray]]

IndexType = Optional[Union[int, str, np.ndarray,
                                      cp.ndarray, List[int], List[str]]]
