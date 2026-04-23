from typing import Optional, Any, Dict, TypeAlias, Union

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
