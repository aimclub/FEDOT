from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, Union

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum

if TYPE_CHECKING:
    from fedot.extensions.runtime_contracts import ExternalModelImplementation, TransformImplementation

ModelFactory = Callable[..., 'ExternalModelImplementation']
TransformFactory = Callable[..., 'TransformImplementation']


class ArrayBackend(Enum):
    numpy = 'numpy'
    torch = 'torch'


class OperationKind(Enum):
    model = 'model'
    transform = 'transform'


@dataclass(frozen=True)
class ModelHyperparamsSchema:
    required: Tuple[str, ...] = ()
    optional: Tuple[str, ...] = ()
    defaults: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCapabilities:
    tasks: Tuple[TaskTypesEnum, ...]
    data_types: Tuple[DataTypesEnum, ...]
    tags: Tuple[str, ...] = ()
    supports_multimodal: bool = False
    backend: ArrayBackend = ArrayBackend.numpy
    output_data_type: Optional[DataTypesEnum] = None
    requires_target: bool = True


@dataclass(frozen=True)
class TransformCapabilities:
    tasks: Tuple[TaskTypesEnum, ...]
    data_types: Tuple[DataTypesEnum, ...]
    output_data_type: DataTypesEnum
    tags: Tuple[str, ...] = ()
    supports_multimodal: bool = False
    backend: ArrayBackend = ArrayBackend.numpy
    requires_fit: bool = True
    requires_target: bool = False


@dataclass(frozen=True)
class ExternalModelSpec:
    name: str
    factory: ModelFactory
    capabilities: ModelCapabilities
    hyperparams_schema: ModelHyperparamsSchema = field(
        default_factory=ModelHyperparamsSchema)
    description: str = ''


@dataclass(frozen=True)
class ExternalTransformSpec:
    name: str
    factory: TransformFactory
    capabilities: TransformCapabilities
    hyperparams_schema: ModelHyperparamsSchema = field(
        default_factory=ModelHyperparamsSchema)
    description: str = ''


ExternalOperationSpec = Union[ExternalModelSpec, ExternalTransformSpec]


@dataclass(frozen=True)
class ExtensionManifest:
    name: str
    version: str
    models: Tuple[ExternalModelSpec, ...] = ()
    module: Optional[str] = None
    description: str = ''
    transforms: Tuple[ExternalTransformSpec, ...] = ()


@dataclass(frozen=True)
class ExtensionError:
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    cause: Optional[Exception] = field(default=None, compare=False, repr=False)


class ExtensionContractError(ValueError):
    """Exception boundary retaining the structured validation or runtime failure."""

    def __init__(self, error: ExtensionError):
        self.error = error
        self.code = error.code
        super().__init__(error.message)


def unwrap_extension_result(result):
    if result.is_left():
        error = result.monoid[0]
        raise ExtensionContractError(error) from error.cause
    return result.value


@dataclass(frozen=True)
class RegisteredExtension:
    manifest: ExtensionManifest
