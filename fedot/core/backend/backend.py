from contextlib import contextmanager
import re
import threading
from typing import Any
import logging
import torch


logger = logging.getLogger(__name__)

_CUDA_DEVICE_PATTERN = re.compile(r'^cuda:\d+$')


class Backend:
    """Singleton object for managing the compute backend (CPU/GPU).

    This class is intended for centralized selection of the libraries/frameworks
    used during data processing:
    - CPU: `numpy` / `pandas` and a `torch.device("cpu")` device
    - GPU: `cupy` / `cudf` and a `torch.device("cuda")` or `torch.device("cuda:N")` device

    Backend state is stored on the instance via:
    - `xp`: an array module (NumPy-like for CPU, CuPy-like for GPU)
    - `pd`: a dataframe module (Pandas-like for CPU, cuDF-like for GPU)
    - `device`: the current PyTorch device
    - `name`: the normalized backend name (`"cpu"`, `"gpu"`, or `"cuda:N"`)

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
    GPU_ALIASES = frozenset({'gpu', 'cuda'})

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
        return 'cpu, gpu, cuda, cuda:<device_index>'

    @classmethod
    def normalize_name(cls, name: Any) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f'Backend name must be a non-empty string, got {name!r}. '
                f'Expected one of: {cls.supported_name_hint()}'
            )

        normalized = name.strip().lower()
        if normalized == cls.DEFAULT_NAME:
            return cls.DEFAULT_NAME
        if normalized in cls.GPU_ALIASES:
            return 'gpu'
        if _CUDA_DEVICE_PATTERN.match(normalized):
            return normalized

        raise ValueError(
            f'Unsupported backend name: {name!r}. '
            f'Expected one of: {cls.supported_name_hint()}'
        )

    @classmethod
    def is_gpu_name(cls, name: str) -> bool:
        return cls.normalize_name(name) != cls.DEFAULT_NAME

    @classmethod
    def torch_device_for_name(cls, name: str) -> torch.device:
        normalized_name = cls.normalize_name(name)
        if normalized_name == cls.DEFAULT_NAME:
            return torch.device('cpu')
        if normalized_name == 'gpu':
            return torch.device('cuda')
        return torch.device(normalized_name)

    def _set_backend(self, name: str):
        normalized_name = self.normalize_name(name)
        if normalized_name == self.DEFAULT_NAME:
            import numpy as xp
            import pandas as pd

            self.xp = xp
            self.pd = pd
            self.device = torch.device("cpu")
            self.name = self.DEFAULT_NAME
            return

        try:
            import cupy as xp
            import cudf as pd
        except Exception as e:
            raise RuntimeError("Can't import cupy or cudf") from e

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.xp = xp
        self.pd = pd
        self.device = self.torch_device_for_name(normalized_name)
        self.name = normalized_name

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
