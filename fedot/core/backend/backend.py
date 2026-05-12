from contextlib import contextmanager
import threading
from typing import Any
import logging
import torch


logger = logging.getLogger(__name__)


def torch_to_xp(tensor: torch.Tensor, xp):
    """
    Convert torch.Tensor to xp array (numpy or cupy).

    Parameters
    ----------
    tensor : torch.Tensor
    xp : module (numpy or cupy)

    Returns
    -------
    xp.ndarray
    """
    if tensor is None:
        return None

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    # numpy backend
    if xp.__name__ == "numpy":
        return tensor.detach().cpu().numpy()

    # cupy backend
    if xp.__name__ == "cupy":
        return xp.fromDlpack(torch.utils.dlpack.to_dlpack(tensor))

    raise ValueError(f"Unsupported xp backend: {xp}")


class Backend:
    """
Singleton object for managing the compute backend (CPU/GPU).

This class is intended for centralized selection of the libraries/frameworks
used during data processing:
- CPU: `numpy` / `pandas` and a `torch.device("cpu")` device
- GPU: `cupy` / `cudf` and a `torch.device("cuda")` device (when CUDA is available)

Backend state is stored on the instance via:
- `xp`: an array module (NumPy-like for CPU, CuPy-like for GPU)
- `pd`: a dataframe module (Pandas-like for CPU, cuDF-like for GPU)
- `device`: the current PyTorch device
- `name`: the backend name (`"cpu"` or `"gpu"`), default is `"cpu"`

Thread-safety is provided: backend switching is synchronized with locks, and
temporary overrides are available through the :meth:`override` context manager.

Examples
--------
Example backend installation (switching):
    backend_name = "gpu"
    backend.set(backend_name)

Example temporary override:
    with backend.override("cpu"):
        # inside this block the backend uses CPU libraries and `torch.device("cpu")`
        ...
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, name: str = "cpu"):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name: str = "cpu"):
        if getattr(self, "_initialized", False):
            return

        self._lock = threading.RLock()
        self.xp: Any = None
        self.pd: Any = None
        self.device: torch.device = None
        self.name: str = "cpu"

        self._set_backend(name)
        self._initialized: bool = True

    def _set_backend(self, name: str):
        if name == "gpu":
            try:
                import cupy as xp
                import cudf as pd
            except Exception as e:
                raise RuntimeError("Can't import cupy or cudf") from e

            self.xp = xp
            self.pd = pd
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise RuntimeError("CUDA is not available")
            self.name = "gpu"
        else:
            import numpy as xp
            import pandas as pd

            self.xp = xp
            self.pd = pd
            self.device = torch.device("cpu")
            self.name = "cpu"

    def set(self, name: str = "cpu"):
        with self._lock:
            self._set_backend(name)

    @contextmanager
    def override(self, name: str):
        with self._lock:
            old_name = self.name
            self._set_backend(name)

        try:
            yield self
        finally:
            with self._lock:
                self._set_backend(old_name)

    def reset(self):
        with self._lock:
            self.xp = None
            self.pd = None
            self.device = None
            self.name = None
            self._initialized = False
