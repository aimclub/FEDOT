from typing import List, Optional, TypeAlias, Union
import os

import numpy as np
import pandas as pd
import torch


PathType: TypeAlias = Union[os.PathLike, str]

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import cudf
except ImportError:
    cudf = None

ARRAY_RUNTIME_TYPES = (np.ndarray,)
PANDAS_RUNTIME_TYPES = (pd.DataFrame, pd.Series)

if cp is not None:
    ARRAY_RUNTIME_TYPES = (*ARRAY_RUNTIME_TYPES, cp.ndarray)
    ArrayType: TypeAlias = Optional[Union[np.ndarray, cp.ndarray]]
    IndexType: TypeAlias = Optional[Union[int, str, np.ndarray,
                                        cp.ndarray, List[int], List[str]]]
    TensorLike: TypeAlias = Optional[Union[torch.Tensor, np.ndarray, cp.ndarray]]
else:
    ArrayType: TypeAlias = Optional[np.ndarray]
    IndexType: TypeAlias = Optional[Union[int, str, np.ndarray, List[int], List[str]]]
    TensorLike: TypeAlias = Optional[Union[torch.Tensor, np.ndarray]]

if cudf is not None:
    PANDAS_RUNTIME_TYPES = (*PANDAS_RUNTIME_TYPES, cudf.DataFrame, cudf.Series)
    PandasType: TypeAlias = Union[pd.DataFrame, pd.Series, cudf.DataFrame, cudf.Series]
else:
    PandasType: TypeAlias = Union[pd.DataFrame, pd.Series]
