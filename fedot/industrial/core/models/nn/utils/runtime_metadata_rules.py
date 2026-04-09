"""Pure rules for runtime metadata ownership around NN output containers."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


_OUTPUT_CONTEXT_KEYS = ('num_classes', 'train_dataloader', 'val_dataloader')
_OUTPUT_FIELD_PATHS = {
    'train_dataloader': ('train_dataloader', ('features', 'train_dataloader')),
    'val_dataloader': ('val_dataloader', ('features', 'val_dataloader')),
    'num_classes': ('num_classes', ('features', 'num_classes')),
    'features': ('features', ('features', 'features')),
}


@dataclass(frozen=True)
class RegistryCheckpointContext:
    model_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    fedcore_id: Optional[str] = None


@dataclass(frozen=True)
class OutputCompatibilityContext:
    features: Any = None
    num_classes: Any = None
    train_dataloader: Any = None
    val_dataloader: Any = None


@dataclass(frozen=True)
class OutputRuntimeAttachmentPlan:
    context_attrs: Dict[str, Any]
    metadata_attrs: Dict[str, Any]

    @property
    def attrs(self) -> Dict[str, Any]:
        return {**self.context_attrs, **self.metadata_attrs}


def build_registry_checkpoint_context(model_id: Optional[str] = None,
                                      checkpoint_path: Optional[str] = None,
                                      fedcore_id: Optional[str] = None) -> RegistryCheckpointContext:
    return RegistryCheckpointContext(
        model_id=model_id,
        checkpoint_path=checkpoint_path,
        fedcore_id=fedcore_id,
    )


def _get_output_field_by_path(obj: Any, path: Any) -> Any:
    if isinstance(path, str):
        path = (path,)

    try:
        result = obj
        for attr in path:
            result = getattr(result, attr) if hasattr(result, attr) else None
            if result is None:
                return None
        return result
    except (AttributeError, TypeError):
        return None


def build_output_compatibility_context(input_data: Any) -> OutputCompatibilityContext:
    if input_data is None:
        return OutputCompatibilityContext()

    resolved = {}
    for field_name, paths in _OUTPUT_FIELD_PATHS.items():
        value = None
        for path in paths:
            value = _get_output_field_by_path(input_data, path)
            if value is not None:
                if field_name == 'features':
                    if isinstance(value, np.ndarray):
                        break
                    if hasattr(value, 'features'):
                        value = value.features
                        break
                else:
                    break
        resolved[field_name] = value

    return OutputCompatibilityContext(**resolved)


def build_output_runtime_attachment_plan(compatibility_context: Optional[OutputCompatibilityContext] = None,
                                         checkpoint_context: Optional[RegistryCheckpointContext] = None,
                                         model: Any = None) -> OutputRuntimeAttachmentPlan:
    if compatibility_context is None:
        compatibility_context = OutputCompatibilityContext()

    context_attrs = {}
    for key in _OUTPUT_CONTEXT_KEYS:
        value = getattr(compatibility_context, key)
        if value is not None:
            context_attrs[key] = value

    metadata_attrs = {}
    if model is not None:
        metadata_attrs['model'] = model

    if checkpoint_context is not None:
        if checkpoint_context.checkpoint_path is not None:
            metadata_attrs['checkpoint_path'] = checkpoint_context.checkpoint_path
        if checkpoint_context.model_id is not None:
            metadata_attrs['model_id'] = checkpoint_context.model_id
        if checkpoint_context.fedcore_id is not None:
            metadata_attrs['fedcore_id'] = checkpoint_context.fedcore_id

    return OutputRuntimeAttachmentPlan(
        context_attrs=context_attrs,
        metadata_attrs=metadata_attrs,
    )


def attach_output_runtime_context(output_data: Any,
                                  plan: OutputRuntimeAttachmentPlan) -> Any:
    for attr_name, value in plan.attrs.items():
        setattr(output_data, attr_name, value)
    return output_data
