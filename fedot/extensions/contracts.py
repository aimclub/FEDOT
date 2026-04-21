from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum

ModelFactory = Callable[[Optional[Dict[str, Any]]], Any]


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


@dataclass(frozen=True)
class ExternalModelSpec:
    name: str
    factory: ModelFactory
    capabilities: ModelCapabilities
    hyperparams_schema: ModelHyperparamsSchema = field(default_factory=ModelHyperparamsSchema)
    description: str = ''


@dataclass(frozen=True)
class ExtensionManifest:
    name: str
    version: str
    models: Tuple[ExternalModelSpec, ...]
    module: Optional[str] = None
    protocols: Optional[Dict[str, Callable[..., Any]]] = None
    description: str = ''


@dataclass(frozen=True)
class ExtensionError:
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegisteredExtension:
    manifest: ExtensionManifest
