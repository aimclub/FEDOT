import torch
from typing import Sequence, List, Optional, Tuple


def expand_features_idx_for_flatten(
    features_idx: Optional[Sequence[int]],
    original_shape: Tuple,
) -> Optional[List[int]]:
    """
    Expand feature indices after flattening (features, channels) → (features * channels).

    Args:
        features_idx: original feature indices (before flatten)
        original_shape: original shape

    Returns:
        Expanded indices for flattened representation
    """
    if features_idx is None or len(original_shape) == 2:
        return features_idx
    
    n_channels = original_shape[2]
    expanded = []

    for f in features_idx:
        base = f * n_channels
        for c in range(n_channels):
            expanded.append(base + c)

    return expanded


def flatten_if_needed(x: torch.Tensor) -> torch.Tensor:
    """Run `flatten_if_needed` routine."""
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        return x.reshape(x.shape[0], -1)
    raise ValueError(f"Expected 2D or 3D tensor, got shape={tuple(x.shape)}")


def restore_if_needed(x: torch.Tensor, original_shape) -> torch.Tensor:
    """Run `restore_if_needed` routine."""
    if original_shape is None or len(original_shape) == 2:
        return x
    else:
        return x.reshape(original_shape)
