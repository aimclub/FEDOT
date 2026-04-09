"""Checkpoint registration planning and orchestration rules for trainer shells."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from fedot.industrial.core.models.nn.utils.registry_context_rules import (
    build_resolved_registry_context,
)
from fedot.industrial.core.models.nn.utils.runtime_metadata_rules import (
    RegistryCheckpointContext,
    build_registry_checkpoint_context,
)


@dataclass(frozen=True)
class CheckpointRegistrationRequest:
    fedcore_id: Optional[str]
    stage: Optional[str]
    should_register: bool


def build_checkpoint_registration_request(model_present: bool,
                                          stage: Optional[str] = None,
                                          explicit_fedcore_id: Optional[str] = None,
                                          trainer_fedcore_id: Optional[str] = None,
                                          thread_local_context: Optional[Tuple[Optional[str], Optional[str]]] = None
                                          ) -> CheckpointRegistrationRequest:
    resolved_context = build_resolved_registry_context(
        explicit_fedcore_id=explicit_fedcore_id,
        trainer_fedcore_id=trainer_fedcore_id,
        thread_local_context=thread_local_context,
    )
    return CheckpointRegistrationRequest(
        fedcore_id=resolved_context.fedcore_id,
        stage=stage,
        should_register=bool(model_present and resolved_context.fedcore_id is not None),
    )


def execute_checkpoint_registration(request: CheckpointRegistrationRequest,
                                    model: Any,
                                    register_model: Callable[..., str],
                                    get_checkpoint_path: Callable[[str, str], Optional[str]]
                                    ) -> RegistryCheckpointContext:
    if not request.should_register:
        return build_registry_checkpoint_context(fedcore_id=request.fedcore_id)

    model_id = register_model(
        fedcore_id=request.fedcore_id,
        model=model,
        stage=request.stage,
        delete_model_after_save=False,
    )
    checkpoint_path = get_checkpoint_path(request.fedcore_id, model_id)

    return build_registry_checkpoint_context(
        model_id=model_id,
        checkpoint_path=checkpoint_path,
        fedcore_id=request.fedcore_id,
    )
