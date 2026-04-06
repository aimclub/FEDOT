from dataclasses import dataclass
from typing import Any, Optional
import torch


@dataclass
class PreparedData:
    features: torch.Tensor
    target: Optional[torch.Tensor] = None
    meta: Any = None # some tag to find PreprocessingPlan