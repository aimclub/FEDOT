"""Pure cleanup planning rules for the model registry shell."""

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Tuple


@dataclass(frozen=True)
class SequenceCleanupTarget:
    attr_name: str
    indices: Tuple[int, ...]
    mutable: bool


@dataclass(frozen=True)
class MappingCleanupTarget:
    attr_name: str
    keys: Tuple[Any, ...]


@dataclass(frozen=True)
class DynamicModelCleanupPlan:
    attr_names: Tuple[str, ...] = ()
    sequence_targets: Tuple[SequenceCleanupTarget, ...] = ()
    mapping_targets: Tuple[MappingCleanupTarget, ...] = ()


@dataclass(frozen=True)
class NestedTrainerCleanupTarget:
    trainer_attr: str
    model_attr: Optional[str] = None


@dataclass(frozen=True)
class TrainerCleanupPlan:
    direct_model_attrs: Tuple[str, ...] = ()
    nested_targets: Tuple[NestedTrainerCleanupTarget, ...] = ()


@dataclass(frozen=True)
class CompressorCleanupPlan:
    model_attrs: Tuple[str, ...] = ()
    has_trainer: bool = False
    dynamic_plan: DynamicModelCleanupPlan = field(default_factory=DynamicModelCleanupPlan)


@dataclass(frozen=True)
class RegistryStorageCleanupPlan:
    clear_checkpoint_bytes: bool = False
    target_column: Optional[str] = None


def build_registry_storage_cleanup_plan(columns: Iterable[str]) -> RegistryStorageCleanupPlan:
    normalized_columns = tuple(columns)
    if 'checkpoint_bytes' in normalized_columns:
        return RegistryStorageCleanupPlan(clear_checkpoint_bytes=True, target_column='checkpoint_bytes')
    return RegistryStorageCleanupPlan()


def build_trainer_cleanup_plan(trainer: Any) -> TrainerCleanupPlan:
    direct_model_attrs = tuple(
        attr_name for attr_name in ('model',)
        if getattr(trainer, attr_name, None) is not None
    )

    nested_targets = []
    nested_trainer = getattr(trainer, '_trainer', None)
    if nested_trainer is not None:
        nested_targets.append(
            NestedTrainerCleanupTarget(
                trainer_attr='_trainer',
                model_attr='model' if getattr(nested_trainer, 'model', None) is not None else None,
            )
        )

    return TrainerCleanupPlan(
        direct_model_attrs=direct_model_attrs,
        nested_targets=tuple(nested_targets),
    )


def build_dynamic_model_cleanup_plan(obj: Any, module_type: type) -> DynamicModelCleanupPlan:
    attr_names = []
    sequence_targets = []
    mapping_targets = []

    for attr_name in dir(obj):
        if attr_name.startswith('__'):
            continue

        try:
            attr = getattr(obj, attr_name, None)
        except Exception:
            continue

        if isinstance(attr, module_type):
            attr_names.append(attr_name)
            continue

        if isinstance(attr, list):
            indices = tuple(index for index, item in enumerate(attr) if isinstance(item, module_type))
            if indices:
                sequence_targets.append(SequenceCleanupTarget(attr_name=attr_name, indices=indices, mutable=True))
            continue

        if isinstance(attr, tuple):
            indices = tuple(index for index, item in enumerate(attr) if isinstance(item, module_type))
            if indices:
                sequence_targets.append(SequenceCleanupTarget(attr_name=attr_name, indices=indices, mutable=False))
            continue

        if isinstance(attr, dict):
            keys = tuple(key for key, value in attr.items() if isinstance(value, module_type))
            if keys:
                mapping_targets.append(MappingCleanupTarget(attr_name=attr_name, keys=keys))

    return DynamicModelCleanupPlan(
        attr_names=tuple(attr_names),
        sequence_targets=tuple(sequence_targets),
        mapping_targets=tuple(mapping_targets),
    )


def build_compressor_cleanup_plan(compressor_object: Any,
                                  model_attrs_to_clean: Iterable[str],
                                  module_type: type) -> CompressorCleanupPlan:
    model_attrs = tuple(
        attr_name for attr_name in model_attrs_to_clean
        if getattr(compressor_object, attr_name, None) is not None
    )

    return CompressorCleanupPlan(
        model_attrs=model_attrs,
        has_trainer=getattr(compressor_object, 'trainer', None) is not None,
        dynamic_plan=build_dynamic_model_cleanup_plan(compressor_object, module_type),
    )
