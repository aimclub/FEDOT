from typing import Optional, Union, TypeAlias, List
import cupy as cp
import numpy as np
import torch
from dataclasses import dataclass

from golem.utilities.data_structures import ComparableEnum as Enum


class StateEnum(Enum):
    FIT = 'fit'
    PREDICT = 'predict'


class TSOrientationEnum(Enum):
    wide = 'wide'
    long = 'long'
