from contextlib import contextmanager
import threading
from typing import Any
import torch

from fedot.core.backend.schemas import (
    BACKEND_CUDA_DEVICE_PATTERN,
    backend_supported_names_hint,
    validate_backend_name,
)

class Backend:
    """Singleton object for managing the compute backend (CPU/GPU).

    This class is intended for centralized selection of the libraries/frameworks
    used during data processing:
    - CPU: `numpy` / `pandas` and a `torch.device("cpu")` device
    - CUDA GPU: `cupy` / `cudf` and a `torch.device("cuda")` or `torch.device("cuda:N")` device
    - MPS GPU: `numpy` / `pandas` and a `torch.device("mps")` device (Apple Silicon)

    Backend state is stored on the instance via:
    - `xp`: an array module (NumPy-like for CPU, CuPy-like for GPU)
    - `pd`: a dataframe module (Pandas-like for CPU, cuDF-like for GPU)
    - `device`: the current PyTorch device
    - `name`: the normalized backend name (`"cpu"`, `"gpu"`, `"mps"`, or `"cuda:N"`)

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
    DEFAULT_NAME = 'cpu'
    CUDA_STACK_NAME = 'gpu'
    CUDA_NAME = 'cuda'
    MPS_NAME = 'mps'
    BACKEND_STRATEGIES = {}

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, name: str = DEFAULT_NAME):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name: str = DEFAULT_NAME):
        if getattr(self, "_initialized", False):
            return

        self._lock = threading.RLock()
        self.xp: Any = None
        self.pd: Any = None
        self.device: torch.device = None
        self.name: str = self.DEFAULT_NAME

        self._set_backend(name)
        self._initialized: bool = True

    @classmethod
    def supported_name_hint(cls) -> str:
        return backend_supported_names_hint()

    @classmethod
    def is_cuda_stack_compatible(cls) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            import cupy  # noqa: F401
            import cudf  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def is_mps_compatible(cls) -> bool:
        mps_backend = getattr(torch.backends, 'mps', None)
        return mps_backend is not None and mps_backend.is_available()

    @classmethod
    def detect_compatible_gpu_backend(cls) -> str:
        if cls.is_cuda_stack_compatible():
            return cls.CUDA_STACK_NAME
        if cls.is_mps_compatible():
            return cls.MPS_NAME
        raise RuntimeError(
            'No FEDOT-compatible GPU backend is available. '
            'CUDA requires NVIDIA GPU with cupy and cudf installed; '
            'MPS requires Apple Silicon with a PyTorch build that supports MPS.'
        )

    @classmethod
    def resolve_name(cls, name: str) -> str:
        normalized_name = cls.normalize_name(name)
        if normalized_name == cls.CUDA_STACK_NAME:
            return cls.detect_compatible_gpu_backend()
        if normalized_name == cls.CUDA_NAME:
            if not cls.is_cuda_stack_compatible():
                raise RuntimeError(
                    "CUDA stack is not available. Install cupy and cudf and ensure "
                    "CUDA is available."
                )
            return cls.CUDA_STACK_NAME
        if BACKEND_CUDA_DEVICE_PATTERN.match(normalized_name):
            if not cls.is_cuda_stack_compatible():
                raise RuntimeError(
                    "CUDA stack is not available. Install cupy and cudf and ensure "
                    "CUDA is available."
                )
            return normalized_name
        return normalized_name

    @classmethod
    def normalize_name(cls, name: Any) -> str:
        return validate_backend_name(name)

    def _set_cpu_backend(self):
        import numpy as xp
        import pandas as pd

        self.xp = xp
        self.pd = pd
        self.device = torch.device("cpu")
        self.name = self.DEFAULT_NAME

    def _set_mps_backend(self):
        if not type(self).is_mps_compatible():
            raise RuntimeError("MPS is not available")

        import numpy as xp
        import pandas as pd

        self.xp = xp
        self.pd = pd
        self.device = torch.device("mps")
        self.name = self.MPS_NAME

    def _set_cuda_stack_backend(self, normalized_name: str):
        if not type(self).is_cuda_stack_compatible():
            raise RuntimeError(
                "CUDA stack is not available. Install cupy and cudf and ensure "
                "CUDA is available, or use backend_name='mps' on Apple Silicon."
            )

        import cupy as xp
        import cudf as pd

        self.xp = xp
        self.pd = pd
        device_name = self.CUDA_NAME if normalized_name == self.CUDA_STACK_NAME else normalized_name
        self.device = torch.device(device_name)
        self.name = normalized_name

    def _set_auto_gpu_backend(self):
        resolved_name = self.detect_compatible_gpu_backend()
        if resolved_name == self.MPS_NAME:
            self._set_mps_backend()
            return
        self._set_cuda_stack_backend(self.CUDA_STACK_NAME)

    def _set_backend(self, name: str):
        normalized_name = self.normalize_name(name)
        if BACKEND_CUDA_DEVICE_PATTERN.match(normalized_name):
            self._set_cuda_stack_backend(normalized_name)
            return

        strategy = self.BACKEND_STRATEGIES.get(normalized_name)
        if strategy is None:
            raise RuntimeError(f'Unsupported backend name after normalization: {normalized_name!r}')

        strategy(self)

    def set(self, name: str = DEFAULT_NAME):
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


Backend.BACKEND_STRATEGIES = {
    Backend.DEFAULT_NAME: Backend._set_cpu_backend,
    Backend.CUDA_STACK_NAME: Backend._set_auto_gpu_backend,
    Backend.CUDA_NAME: lambda backend: backend._set_cuda_stack_backend(Backend.CUDA_STACK_NAME),
    Backend.MPS_NAME: Backend._set_mps_backend,
}


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
