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


class ScalingMethodEnum(Enum):
    min_max = "min_max"
    standard = "standard"
    robust = "robust"
    seasonal = "seasonal"
    rolling = "rolling"


class FilteringMethodEnum(Enum):
    quantile = "quantile"


@dataclass
class PreprocessingStep:
    step: PreprocessingStepEnum
    method: Union[EmbeddingMethodEnum, ImputationMethodEnum, ScalingMethodEnum]
    features_idx: IndexType
    state: StateEnum = StateEnum.FIT
    step_args: dict[str, Any] = field(default_factory=dict)
    # TODO: to args all model_name...
    model_name: Optional[str] = None 
    batch_size: Optional[int] = None
    device: torch.device = torch.device("cpu")
    model_hash: Optional[str] = None
