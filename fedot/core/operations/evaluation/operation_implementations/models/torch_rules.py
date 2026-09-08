from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TorchTabularFitPlan:
    """Resolved, deterministic training configuration for a Torch tabular model."""

    device: str
    hidden_layer_sizes: tuple[int, ...]
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    validation_fraction: float
    patience: int
    random_state: int


def resolve_torch_device(
    requested_device: str,
    input_device: str,
    cuda_available: bool,
) -> str:
    """Resolve model device while preserving the TensorData runtime by default."""
    if requested_device == 'auto':
        return input_device
    if requested_device.startswith('cuda') and not cuda_available:
        raise ValueError(
            "device='cuda' was requested, but CUDA is unavailable")
    if requested_device == 'cuda' and input_device.startswith('cuda:'):
        return input_device
    if requested_device != 'cpu' and requested_device != 'cuda' and not requested_device.startswith('cuda:'):
        raise ValueError(f"Unsupported Torch device: {requested_device!r}")
    return requested_device


def normalize_hidden_layer_sizes(value: Any) -> tuple[int, ...]:
    """Normalize sklearn-compatible hidden-layer notation to an immutable tuple."""
    if value is None:
        return ()
    if isinstance(value, int):
        sizes = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sizes = tuple(int(size) for size in value)
    else:
        raise ValueError(
            'hidden_layer_sizes must be an int or a sequence of ints')
    if any(size <= 0 for size in sizes):
        raise ValueError('hidden_layer_sizes values must be positive')
    return sizes


def build_torch_tabular_fit_plan(
    params: Mapping[str, Any],
    samples_count: int,
    input_device: str,
    cuda_available: bool,
    dataloader_kwargs: Mapping[str, Any],
) -> TorchTabularFitPlan:
    """Build the complete fit plan at the model boundary."""
    requested_batch_size = params.get(
        'batch_size', dataloader_kwargs.get('batch_size', 32))
    batch_size = min(samples_count, int(requested_batch_size))
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')

    validation_fraction = float(params.get('validation_fraction', 0.1))
    if not 0 <= validation_fraction < 1:
        raise ValueError('validation_fraction must be in [0, 1)')

    epochs = int(params.get('epochs', 200))
    patience = int(params.get('patience', 20))
    learning_rate = float(params.get('learning_rate', 0.01))
    if epochs <= 0:
        raise ValueError('epochs must be positive')
    if patience <= 0:
        raise ValueError('patience must be positive')
    if learning_rate <= 0:
        raise ValueError('learning_rate must be positive')

    return TorchTabularFitPlan(
        device=resolve_torch_device(
            requested_device=str(params.get('device', 'auto')),
            input_device=input_device,
            cuda_available=cuda_available,
        ),
        hidden_layer_sizes=normalize_hidden_layer_sizes(
            params.get('hidden_layer_sizes')),
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=float(params.get('weight_decay', 0.0001)),
        validation_fraction=validation_fraction,
        patience=patience,
        random_state=int(params.get('random_state', 42)),
    )
