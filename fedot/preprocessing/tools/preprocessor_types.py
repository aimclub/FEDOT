from typing import Optional, Any, Dict, TypeAlias, Union

from dataclasses import dataclass, field
from golem.utilities.data_structures import ComparableEnum as Enum
import torch
from fedot.core.data.complex_types import IndexType
from fedot.core.data.tools import StateEnum

class EmbeddingMethodEnum(Enum):
    transformer = "sentence_transformer"


@dataclass
class EmbedderParameters:
    method: EmbeddingMethodEnum
    model_name: str
    batch_size: int
    device: torch.device


class EncodingStrategyEnum(Enum):
    label = "label"
    ohe = "ohe"


@dataclass
class CategoricalEncodingDecision:
    categorical_columns: IndexType
    strategy: Optional[EncodingStrategyEnum] = None
    encoder: Any = None


EncodingStrategyType: TypeAlias = Optional[Union[Dict, CategoricalEncodingDecision]]


class PreprocessingStepEnum(Enum):
    encoding = "encoding"
    embedding = "embedding"
    imputation = "imputation"
    scaling = "scaling"
    filtering = "filtering"
    image_preprocessing = "image_preprocessing"


class EncodingMethodEnum(Enum):
    label = "label"
    ohe = "ohe"
    target_encoding = "target_encoding"


class ImputationMethodEnum(Enum):
    mean = "mean"
    median = "median"
    mode = "mode"
    constant = "constant"
    delete_raw = "delete_raw"
    ts_mean = "ts_mean"
    ts_median = "ts_median"
    ts_constant = "ts_constant"
    ts_fill = "ts_fill"
    ts_rolling = "ts_rolling"
    ts_kalman = "ts_kalman"
    ts_linear_inter = "ts_linear_interpolation"
    ts_polynomial_inter = "ts_polynomial_interpolation"
    ts_spline_inter = "ts_spline_interpolation"


class ScalingMethodEnum(Enum):
    min_max = "min_max"
    standard = "standard"
    robust = "robust"
    seasonal = "seasonal"
    rolling = "rolling"
    standart_per_channel = "standart_per_channel"


class ImagePreprocessingMethodEnum(Enum):
    contrast_equalization = "contrast_equalization"
    contrast_stretching = "contrast_stretching"
    gamma_correction = "gamma_correction"
    log_transformation = "log_transformation"


class FilteringMethodEnum(Enum):
    quantile = "quantile"


@dataclass
class PreprocessingStep:
    step: PreprocessingStepEnum
    method: Union[EmbeddingMethodEnum, ImputationMethodEnum, ScalingMethodEnum]
    features_idx: IndexType
    state: StateEnum = StateEnum.FIT
    step_args: dict[str, Any] = field(default_factory=dict)
    model_hash: Optional[str] = None
