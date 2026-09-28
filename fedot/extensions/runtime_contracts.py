"""Array-level extension boundary, independent of TensorData lifecycle."""
from dataclasses import dataclass
from typing import Optional, Protocol, Union

import numpy as np
import torch

Array = Union[np.ndarray, torch.Tensor]


class ModelImplementation(Protocol):
    def fit(self, features: Array, target: Optional[Array]) -> object: ...

    def predict(self, features: Array) -> Array: ...


class ProbabilisticModelImplementation(ModelImplementation, Protocol):
    """Model implementation with optional probability prediction support."""

    def predict_proba(self, features: Array) -> Array: ...


ExternalModelImplementation = ModelImplementation


class TransformImplementation(Protocol):
    def transform(self, features: Array) -> Array: ...


class FittableTransformImplementation(TransformImplementation, Protocol):
    def fit(self, features: Array, target: Optional[Array] = None) -> object: ...


@dataclass(frozen=True)
class ModelInput:
    features: Array
    target: Optional[Array] = None
    idx: Optional[Array] = None


@dataclass(frozen=True)
class TransformInput:
    features: Array
    target: Optional[Array] = None
    idx: Optional[Array] = None


@dataclass(frozen=True)
class ModelOutput:
    prediction: Array


@dataclass(frozen=True)
class TransformOutput:
    features: Array
