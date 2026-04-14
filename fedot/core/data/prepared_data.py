from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Tuple
import torch


@dataclass
class PreparedData:
    features: torch.Tensor
    target: Optional[torch.Tensor] = None
    idx_mapping: Dict[int, int] = field(default_factory=dict)
    new_cols_dict: Optional[dict[int, int]] = None
    ts_shape: Optional[Tuple[int, int]] = None