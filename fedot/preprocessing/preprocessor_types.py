from typing import Optional, Any, Dict, TypeAlias, Union

from dataclasses import dataclass
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


class EncodingMethodEnum(Enum):
    label = "label"
    ohe = "ohe"
    target_encoding = "target_encoding"


class ImputationMethodEnum(Enum):
    simple = "simple"
    moda = "moda"
    float_nan = "float_nan"


class ScalingMethodEnum(Enum):
    min_max = "min_max"
    standard = "standard"


@dataclass
class PreprocessingStep:
    step: PreprocessingStepEnum
    method: Union[EmbeddingMethodEnum, ImputationMethodEnum, ScalingMethodEnum]
    features_idx: IndexType
    state: StateEnum = StateEnum.FIT
    model_name: Optional[str] = None 
    batch_size: Optional[int] = None
    device: torch.device = torch.device("cpu")
    model_hash: Optional[str] = None
