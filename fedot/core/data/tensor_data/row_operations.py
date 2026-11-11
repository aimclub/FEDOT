"""Row selection and alignment without moving tensor labels to host memory."""
from typing import Sequence

import numpy as np
import torch

from fedot.core.backend.array_conversions import as_numpy

def take_rows(value, positions: torch.Tensor):
    """Select aligned rows; host indices are needed only for host-backed fields."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.index_select(0, positions.to(value.device))
    positions = as_numpy(positions)
    if isinstance(value, np.ndarray):
        return value[positions]
    return [value[int(position)] for position in positions]


def common_row_indices(indices: Sequence[torch.Tensor]) -> torch.Tensor:
    """Intersect tensor row labels; preserve one branch's order, sort intersections."""
    common = indices[0]
    if len(indices) > 1:
        common = torch.unique(common, sorted=True)
        for labels in indices[1:]:
            common = common[torch.isin(common, labels.to(common.device))]
    return common


def row_positions(labels: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Locate a known subset of labels, taking the last row for duplicate labels."""
    order = torch.argsort(labels, stable=True)
    positions = torch.searchsorted(labels[order], selected.to(labels.device), right=True) - 1
    return order[positions]
