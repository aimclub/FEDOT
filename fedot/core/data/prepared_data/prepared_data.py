from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Tuple
import torch
import numpy as np


@dataclass
class PreparedData:
    features: torch.Tensor
    target: Optional[torch.Tensor] = None
    row_idx: Any = field(default=None, kw_only=True)
    categorical_features: Optional[np.ndarray] = field(default=None, kw_only=True)
    categorical_source_idx: list[int] = field(default_factory=list, kw_only=True)

    idx_mapping: Dict[int, int] = field(default_factory=dict)
    new_cols_dict: Optional[dict[int, int]] = None
    retained_idx: Optional[list[int]] = field(default=None, kw_only=True)
    ts_shape: Optional[Tuple[int, int]] = None
