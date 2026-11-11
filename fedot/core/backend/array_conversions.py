"""Explicit array exchange, independent of the global FEDOT Backend.

Keep dtype and share storage by default. Detach Torch inputs at external data
boundaries; request copy=True for private buffers and pass devices explicitly.
CuPy is imported only when a CUDA array conversion is requested.
"""
from typing import Any, Optional, Union

import numpy as np
import torch


def as_numpy(value: Any, *, dtype=None, copy: bool = False) -> Optional[np.ndarray]:
    """Return a CPU array; CUDA input requires a device-to-host transfer."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        host = value.detach().cpu().numpy()
        copy = copy and value.device.type == 'cpu'
    elif hasattr(value, '__cuda_array_interface__'):
        host = as_cupy(value).get()
        copy = False  # A device-to-host transfer already owns independent storage.
    else:
        host = value
    return np.array(host, dtype=dtype, copy=True) if copy else np.asarray(host, dtype=dtype)


def as_torch(value: Any, *, device: Optional[Union[str, torch.device]] = None,
             dtype: Optional[torch.dtype] = None, copy: bool = False,
             contiguous: bool = False, detach: bool = True) -> Optional[torch.Tensor]:
    """Return detached Torch data, preserving the source device unless specified.

    NumPy/host sequences start on CPU; CuPy exchanges CUDA storage via DLPack.
    Transfers, dtype changes, copy=True and contiguous=True can allocate memory.
    detach=False preserves a Torch input's autograd graph, e.g. for extensions.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach() if detach else value
    elif hasattr(value, '__cuda_array_interface__'):
        tensor = torch.from_dlpack(value)
    else:
        tensor = torch.as_tensor(value, device='cpu')
    tensor = tensor.to(device=device, dtype=dtype, copy=copy)
    return tensor.contiguous() if contiguous else tensor


def as_cupy(value: Any, *, device: Optional[Union[str, torch.device]] = None,
            dtype=None, copy: bool = False, contiguous: bool = False):
    """Return CUDA data; preserve CUDA source devices unless a device is given."""
    if value is None:
        return None
    # Optional GPU dependency: CPU-only users do not need CuPy installed.
    import cupy

    if device is None:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            device = value.device
        elif hasattr(value, '__cuda_array_interface__'):
            device = torch.device('cuda', value.device.id)
        else:
            device = torch.device('cuda', cupy.cuda.runtime.getDevice())
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('CuPy arrays require a CUDA device')
    ordinal = device.index if device.index is not None else cupy.cuda.runtime.getDevice()
    with cupy.cuda.Device(ordinal):
        if isinstance(value, torch.Tensor):
            array = cupy.from_dlpack(as_torch(value, device=device, copy=copy))
            array = array.astype(dtype, copy=False) if dtype is not None else array
        else:
            array = cupy.array(value, dtype=dtype, copy=copy)
        return cupy.ascontiguousarray(array) if contiguous else array


def torch_to_xp(tensor: torch.Tensor, xp):
    """Exchange a preprocessing tensor with the explicitly selected array module."""
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'Expected torch.Tensor, got {type(tensor)}')
    if xp.__name__ == 'numpy':
        return as_numpy(tensor)
    if xp.__name__ == 'cupy':
        return as_cupy(tensor)
    raise ValueError(f'Unsupported xp backend: {xp}')
