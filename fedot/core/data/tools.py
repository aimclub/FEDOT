from typing import Optional, Union, TypeAlias, List
import cupy as cp
import numpy as np

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task


IndexType: TypeAlias = Optional[Union[int, str, np.ndarray, 
                                      cp.ndarray, List[int], List[str]]]
TaskType: TypeAlias = Optional[Union[Task, str]]
DataType: TypeAlias = Optional[Union[DataTypesEnum, str]]