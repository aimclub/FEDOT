"""Device-local row indices for stratified tensor hold-outs."""
from typing import Optional, Tuple

import torch


def stratified_split_indices(target: torch.Tensor, train_size: int,
                             random_seed: int = 42) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Allocate rows proportionally, using a local generator on the target device.

    Multioutput rows are treated as joint strata. Return None when strata cannot
    support both subsets; the caller keeps its existing non-stratified policy.
    Seeds reproduce splits on the same device, not sklearn's random permutations.
    """
    count = len(target)
    _, groups, counts = torch.unique(target, dim=0, return_inverse=True, return_counts=True)
    if min(train_size, count - train_size) < len(counts) or bool((counts < 2).any()):
        return None
    device = target.device
    generator = torch.Generator(device=device).manual_seed(random_seed)

    # Largest-remainder allocation: exact total size, random tie breaking.
    numerators = counts * train_size
    allocation = numerators // count
    remainder = train_size - allocation.sum()
    ties = torch.randperm(len(counts), device=device, generator=generator)
    priority = ties[torch.argsort((numerators % count)[ties], descending=True, stable=True)]
    allocation[priority] += (torch.arange(len(counts), device=device) < remainder).to(torch.long)

    shuffled = torch.randperm(count, device=device, generator=generator)
    grouped = shuffled[torch.argsort(groups[shuffled], stable=True)]
    group = groups[grouped]
    rank = torch.arange(count, device=device) - (counts.cumsum(0) - counts)[group]
    train_offsets = allocation.cumsum(0) - allocation
    test_counts = counts - allocation
    test_offsets = test_counts.cumsum(0) - test_counts
    # Scatter into fixed-size subsets avoids extracting GPU counts to Python.
    destination = torch.where(rank < allocation[group], train_offsets[group] + rank,
                              train_size + test_offsets[group] + rank - allocation[group])
    indices = torch.empty(count, device=device, dtype=torch.long)
    indices.scatter_(0, destination, grouped)
    train = indices[:train_size][torch.randperm(train_size, device=device, generator=generator)]
    test = indices[train_size:][torch.randperm(count - train_size, device=device, generator=generator)]
    return train, test
