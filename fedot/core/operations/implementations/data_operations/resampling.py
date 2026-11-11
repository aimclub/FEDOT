from dataclasses import replace
from typing import Optional

import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.base import TensorDataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class ResampleImplementation(TensorDataOperationImplementation):
    """Binary class balancing by device-local row sampling.

    Preserve the historical expand_minority/reduce_majority policy, ratio
    correction, automatic replacement and minority-first output order. Random
    draws use Torch rather than NumPy; seeds reproduce draws on the same device,
    not sklearn's random sequence. Inference leaves the sample set unchanged.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.n_samples = None
        self.random_state = None
        self._sampling_seed = None

    @property
    def balance(self) -> str:
        return self.params.setdefault('balance', 'expand_minority')

    @property
    def replace(self) -> bool:
        return self.params.setdefault('replace', False)

    @property
    def balance_ratio(self) -> float:
        return self.params.setdefault('balance_ratio', 1.0)

    def fit(self, data: TensorData):
        # ImplementationRandomStateHandler supplies the pipeline fitting seed.
        self._sampling_seed = self.random_state
        return self

    def transform(self, data: TensorData) -> TensorData:
        return replace(data, predict=None)

    def transform_for_fit(self, data: TensorData) -> TensorData:
        if data.target.numel() != len(data.features):
            raise ValueError('Resample supports a single classification target per row.')
        target = data.target.reshape(-1)
        classes, counts = torch.unique(target, sorted=True, return_counts=True)
        if len(classes) != 2 or bool(counts[0] == counts[1]):
            return self.transform(data)
        minority_class = classes[counts.argmin()]
        minority = torch.nonzero(target == minority_class, as_tuple=True)[0]
        majority = torch.nonzero(target != minority_class, as_tuple=True)[0]
        # Historical behavior: None, zero and out-of-range ratios mean full balance.
        if not self.balance_ratio or not 0 <= self.balance_ratio <= 1:
            self.params.update(balance_ratio=1)
        difference = len(majority) - len(minority)
        if self.balance == 'expand_minority':
            selected = minority
            self.n_samples = round(len(minority) + difference * self.balance_ratio)
        elif self.balance == 'reduce_majority':
            selected = majority
            self.n_samples = round(len(majority) - difference * self.balance_ratio)
        else:
            raise ValueError(f'Unknown resampling balance strategy: {self.balance}')
        if not self.replace and self.n_samples >= len(selected):
            self.params.update(replace=True)
        generator = None
        if self._sampling_seed is not None:
            generator = torch.Generator(device=data.device).manual_seed(self._sampling_seed)
        if self.replace:
            offsets = torch.randint(len(selected), (self.n_samples,), device=data.device, generator=generator)
        else:
            offsets = torch.randperm(len(selected), device=data.device, generator=generator)[:self.n_samples]
        sampled = selected[offsets]
        positions = (torch.cat((sampled, majority)) if self.balance == 'expand_minority'
                     else torch.cat((minority, sampled)))
        output = data.subset_rows(positions)
        # Keep the existing fresh positional labels: resampled rows may repeat,
        # while downstream branch alignment requires distinct output row labels.
        return replace(output, predict=None, idx=torch.arange(len(positions), device=data.device))
