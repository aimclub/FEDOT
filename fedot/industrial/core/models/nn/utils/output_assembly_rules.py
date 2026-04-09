"""Helpers for assembling prediction output containers with runtime metadata."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from fedot.industrial.core.models.nn.utils.runtime_metadata_rules import (
    OutputCompatibilityContext,
    RegistryCheckpointContext,
    attach_output_runtime_context,
    build_output_runtime_attachment_plan,
)


@dataclass(frozen=True)
class OutputContainerRequest:
    base_kwargs: Dict[str, Any]


def build_output_container_request(features: Any,
                                   task: Any,
                                   predict: Any,
                                   data_type: Any,
                                   **extra_kwargs: Any) -> OutputContainerRequest:
    base_kwargs = {
        'features': features,
        'task': task,
        'predict': predict,
        'data_type': data_type,
    }
    base_kwargs.update(extra_kwargs)
    return OutputContainerRequest(base_kwargs=base_kwargs)


def assemble_output_container(factory: Callable[..., Any],
                              request: OutputContainerRequest,
                              compatibility_context: Optional[OutputCompatibilityContext] = None,
                              checkpoint_context: Optional[RegistryCheckpointContext] = None,
                              model: Any = None) -> Any:
    output_data = factory(**request.base_kwargs)
    return attach_output_runtime_context(
        output_data,
        build_output_runtime_attachment_plan(
            compatibility_context=compatibility_context,
            checkpoint_context=checkpoint_context,
            model=model,
        ),
    )
