from dataclasses import dataclass
from typing import Optional, Union, TypeAlias, List

import numpy as np
import torch
import cupy as cp

from golem.utilities.data_structures import ComparableEnum as Enum


class StateEnum(Enum):
    """
    Preprocessing state used by the data pipeline.

    `FIT` means the pipeline is allowed to fit encoders/transformations and to
    infer target/feature structures.
    `PREDICT` means the pipeline should not fit stateful transforms and should
    only transform data for inference.
    """
    FIT = 'fit'
    PREDICT = 'predict'


class TSOrientationEnum(Enum):
    """
    Time-series orientation used during time-series preprocessing.

    - `wide`: each sample contains multiple time steps/features in a wide layout.
    - `long`: each sample is represented in a long/stacked layout.
    """
    wide = 'wide'
    long = 'long'
