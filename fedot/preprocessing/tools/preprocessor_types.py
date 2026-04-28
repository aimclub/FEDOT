from typing import Optional, Any, Dict, TypeAlias, Union

from dataclasses import dataclass, field
from golem.utilities.data_structures import ComparableEnum as Enum
import torch
from fedot.core.data.complex_types import IndexType
from fedot.core.data.tools import StateEnum
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler


class EmbeddingMethodEnum(Enum):
    """Enumeration of embeddingmethod options."""
    transformer = "sentence_transformer"


@dataclass
class EmbedderParameters:
    """EmbedderParameters implementation."""
    method: EmbeddingMethodEnum
    model_name: str
    batch_size: int
    device: torch.device


class EncodingStrategyEnum(Enum):
    """Enumeration of encodingstrategy options."""
    label = "label"
    ohe = "ohe"


@dataclass
class CategoricalEncodingDecision:
    """CategoricalEncodingDecision implementation."""
    categorical_columns: IndexType
    strategy: Optional[EncodingStrategyEnum] = None
    encoder: Any = None


EncodingStrategyType: TypeAlias = Optional[Union[Dict, CategoricalEncodingDecision]]


class PreprocessingStepEnum(Enum):
    """Enumeration of preprocessingstep options."""
    encoding = "encoding"
    embedding = "embedding"
    imputation = "imputation"
    scaling = "scaling"
    filtering = "filtering"
    image_preprocessing = "image_preprocessing"
    custom = "custom"
    target_encoding = "target_encoding"


class EncodingMethodEnum(Enum):
    """Enumeration of encodingmethod options."""
    label = "label"
    ohe = "ohe"
    target_encoding = "target_encoding"


class ImputationMethodEnum(Enum):
    """Enumeration of imputationmethod options."""
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
    """Enumeration of scalingmethod options."""
    min_max = "min_max"
    standard = "standard"
    robust = "robust"
    seasonal = "seasonal"
    rolling = "rolling"
    standart_per_channel = "standart_per_channel"


class ImagePreprocessingMethodEnum(Enum):
    """Enumeration of imagepreprocessingmethod options."""
    contrast_equalization = "contrast_equalization"
    contrast_stretching = "contrast_stretching"
    gamma_correction = "gamma_correction"
    log_transformation = "log_transformation"


class FilteringMethodEnum(Enum):
    """Enumeration of filteringmethod options."""
    quantile = "quantile"


@dataclass
class PreprocessingStep:
    """PreprocessingStep definition used in preprocessing flow."""
    step: PreprocessingStepEnum
    method: Union[Enum, str]
    features_idx: IndexType
    implementation: Optional[AbstractPreprocessingHandler] = None
    state: StateEnum = StateEnum.FIT
    step_args: dict[str, Any] = field(default_factory=dict)
    model_hash: Optional[str] = None
