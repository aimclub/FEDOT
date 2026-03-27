from typing import Union, Optional, TypeAlias, List
import os
import pandas as pd
import numpy as np
import cudf
import cupy as cp


PathType: TypeAlias = Union[os.PathLike, str]

PandasType: TypeAlias = Union[pd.DataFrame, pd.Series, cudf.DataFrame, cudf.Series]

ArrayType: TypeAlias = Optional[Union[np.ndarray, cp.ndarray]]

IndexType: TypeAlias = Optional[Union[int, str, np.ndarray, 
                                      cp.ndarray, List[int], List[str]]]
