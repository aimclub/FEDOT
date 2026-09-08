from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ModelEngine(str, Enum):
    """Concrete library family used to fit and execute a model."""

    CUML = 'cuml'
    TORCH = 'torch'
    SKLEARN = 'sklearn'


class RuntimePlatform(str, Enum):
    LINUX = 'linux'
    WSL = 'wsl'
    WINDOWS = 'windows'
    MACOS = 'macos'
    OTHER = 'other'


@dataclass(frozen=True)
class ModelEngineCapabilities:
    """Side-effect-free snapshot used to choose a model engine."""

    platform: RuntimePlatform
    cuda_available: bool
    installed_engines: frozenset[ModelEngine]


@dataclass(frozen=True)
class ModelEngineRequest:
    """Requested engine and the engines implemented by an operation strategy."""

    supported_engines: tuple[ModelEngine, ...]
    preferred_engine: Optional[ModelEngine] = None
    require_acceleration: bool = False


@dataclass(frozen=True)
class ModelEnginePlan:
    """Resolved model implementation and its execution device."""

    engine: ModelEngine
    device_type: str


def resolve_runtime_platform(system_name: str, release: str = '') -> RuntimePlatform:
    """Normalize OS information and keep WSL distinct from native Linux."""
    normalized_system = system_name.lower()
    normalized_release = release.lower()
    if normalized_system == 'linux' and ('microsoft' in normalized_release or 'wsl' in normalized_release):
        return RuntimePlatform.WSL
    if normalized_system == 'linux':
        return RuntimePlatform.LINUX
    if normalized_system == 'windows':
        return RuntimePlatform.WINDOWS
    if normalized_system == 'darwin':
        return RuntimePlatform.MACOS
    return RuntimePlatform.OTHER


def build_model_engine_plan(
    request: ModelEngineRequest,
    capabilities: ModelEngineCapabilities,
) -> ModelEnginePlan:
    """Select the first usable engine, or validate an explicitly requested one."""
    if not request.supported_engines:
        raise ValueError('At least one supported model engine is required')

    candidates = request.supported_engines
    if request.preferred_engine is not None:
        if request.preferred_engine not in request.supported_engines:
            raise ValueError(
                f'Model engine {request.preferred_engine.value!r} is not implemented by this strategy'
            )
        candidates = (request.preferred_engine,)

    failures = []
    for engine in candidates:
        device_type, failure = _resolve_engine_device(engine, capabilities)
        if failure is not None:
            failures.append(failure)
            continue
        if request.require_acceleration and device_type != 'cuda':
            failures.append(f'{engine.value} is available only on CPU')
            continue
        return ModelEnginePlan(engine=engine, device_type=device_type)

    raise RuntimeError('No compatible model engine is available: ' + '; '.join(failures))


def _resolve_engine_device(
    engine: ModelEngine,
    capabilities: ModelEngineCapabilities,
) -> tuple[Optional[str], Optional[str]]:
    if engine not in capabilities.installed_engines:
        return None, f'{engine.value} is not installed'

    if engine is ModelEngine.CUML:
        if capabilities.platform is not RuntimePlatform.LINUX:
            return None, f'cuML is enabled only on native Linux, got {capabilities.platform.value}'
        if not capabilities.cuda_available:
            return None, 'cuML requires an available CUDA device'
        return 'cuda', None

    if engine is ModelEngine.TORCH:
        return ('cuda' if capabilities.cuda_available else 'cpu'), None

    return 'cpu', None
