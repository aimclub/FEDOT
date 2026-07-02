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
    MPS_NAME = 'mps'
    GENERIC_GPU_ALIASES = frozenset({'gpu'})

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
        return 'cpu, gpu, cuda, mps, cuda:<device_index>'

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
        if normalized_name == 'cuda':
            if not cls.is_cuda_stack_compatible():
                raise RuntimeError(
                    "CUDA stack is not available. Install cupy and cudf and ensure "
                    "CUDA is available."
                )
            return cls.CUDA_STACK_NAME
        if _CUDA_DEVICE_PATTERN.match(normalized_name):
            if not cls.is_cuda_stack_compatible():
                raise RuntimeError(
                    "CUDA stack is not available. Install cupy and cudf and ensure "
                    "CUDA is available."
                )
            return normalized_name
        return normalized_name

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
        if normalized in cls.GENERIC_GPU_ALIASES:
            return cls.CUDA_STACK_NAME
        if normalized == 'cuda':
            return 'cuda'
        if normalized == cls.MPS_NAME:
            return cls.MPS_NAME
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
        resolved_name = cls.resolve_name(name)
        if resolved_name == cls.DEFAULT_NAME:
            return torch.device('cpu')
        if resolved_name == cls.CUDA_STACK_NAME:
            return torch.device('cuda')
        if resolved_name == cls.MPS_NAME:
            return torch.device('mps')
        return torch.device(resolved_name)

    def _set_cpu_backend(self):
        import numpy as xp
        import pandas as pd

        self.xp = xp
        self.pd = pd
        self.device = torch.device("cpu")
        self.name = self.DEFAULT_NAME

    def _set_mps_backend(self):
        if not self.is_mps_compatible():
            raise RuntimeError("MPS is not available")

        import numpy as xp
        import pandas as pd

        self.xp = xp
        self.pd = pd
        self.device = torch.device("mps")
        self.name = self.MPS_NAME

    def _set_cuda_stack_backend(self, normalized_name: str):
        if not self.is_cuda_stack_compatible():
            raise RuntimeError(
                "CUDA stack is not available. Install cupy and cudf and ensure "
                "CUDA is available, or use backend_name='mps' on Apple Silicon."
            )

        import cupy as xp
        import cudf as pd

        self.xp = xp
        self.pd = pd
        self.device = self.torch_device_for_name(normalized_name)
        self.name = normalized_name

    def _set_backend(self, name: str):
        normalized_name = self.normalize_name(name)
        if normalized_name == self.DEFAULT_NAME:
            self._set_cpu_backend()
            return

        if normalized_name == self.CUDA_STACK_NAME:
            resolved_name = self.detect_compatible_gpu_backend()

            if resolved_name == self.MPS_NAME:
                self._set_mps_backend()
                logger.info(
                    "Using MPS with NumPy/Pandas and torch.device('mps'). ",
                    "For cudf/cupy support, CUDA is required.",
                )
                return
            self._set_cuda_stack_backend(self.CUDA_STACK_NAME)
            return

        if normalized_name == 'cuda':
            self._set_cuda_stack_backend(self.CUDA_STACK_NAME)
            logger.info(
                "Using CUDA stack with cudf/cupy and torch.device('cuda'). ",
            )
            return

        if normalized_name == self.MPS_NAME:
            self._set_mps_backend()
            logger.info(
                "Using MPS with NumPy/Pandas and torch.device('mps'). ",
                "For cudf/cupy support, CUDA is required.",
            )
            return

        if _CUDA_DEVICE_PATTERN.match(normalized_name):
            self._set_cuda_stack_backend(normalized_name)
            logger.info(
                f"Using CUDA stack with cudf/cupy and torch.device('{normalized_name}'). ",
            )
            return

        raise RuntimeError(f'Unsupported backend name after normalization: {normalized_name!r}')

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
