"""Pure rules for runtime metadata ownership around NN output containers."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


_OUTPUT_CONTEXT_KEYS = ('num_classes', 'train_dataloader', 'val_dataloader')


@dataclass(frozen=True)
class RegistryCheckpointContext:
    model_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    fedcore_id: Optional[str] = None


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


def build_output_runtime_attachment_plan(extracted_fields: Optional[Mapping[str, Any]] = None,
                                         checkpoint_context: Optional[RegistryCheckpointContext] = None,
                                         model: Any = None) -> OutputRuntimeAttachmentPlan:
    context_attrs = {}
    for key in _OUTPUT_CONTEXT_KEYS:
        value = (extracted_fields or {}).get(key)
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
