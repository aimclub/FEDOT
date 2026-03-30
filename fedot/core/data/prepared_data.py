from dataclasses import dataclass
from typing import Any
import torch


@dataclass
class PreparedData:
    features: torch.Tensor
    meta: Any = None # some tag to find PreprocessingPlan