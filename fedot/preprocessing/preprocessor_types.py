from typing import Optional, Any, Dict, TypeAlias, Union, Callable

from dataclasses import dataclass
from golem.utilities.data_structures import ComparableEnum as Enum
import torch
from fedot.core.data.complex_types import IndexType


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


class ImputationMethodEnum(Enum):
    simple = "simple"
    moda = "moda"


class ScalingMethodEnum(Enum):
    min_max = "min_max"
    standard = "standard"


@dataclass
class PreprocessingStep:
    step: PreprocessingStepEnum
    method: Union[EmbeddingMethodEnum, ImputationMethodEnum, ScalingMethodEnum]
    features_idx: IndexType
    model_hash: Optional[str] = None
