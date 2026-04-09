"""Effect-light orchestration helpers for registry checkpoint persistence."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class RegistryCheckpointTargetPlan:
    checkpoint_path: str
    should_save_file: bool


@dataclass(frozen=True)
class RegistryPersistenceRequest:
    fedcore_id: str
    checkpoint_path: str
    cleanup_after_save: bool
    should_save_file: bool
    record: Dict[str, Any]


def build_registry_checkpoint_target_plan(model_path: Optional[str],
                                          generated_checkpoint_path: str,
                                          model_path_exists: bool) -> RegistryCheckpointTargetPlan:
    if model_path and model_path_exists:
        return RegistryCheckpointTargetPlan(
            checkpoint_path=model_path,
            should_save_file=False,
        )

    return RegistryCheckpointTargetPlan(
        checkpoint_path=generated_checkpoint_path,
        should_save_file=True,
    )


def build_registry_persistence_request(fedcore_id: str,
                                       checkpoint_path: str,
                                       cleanup_after_save: bool,
                                       should_save_file: bool,
                                       record: Dict[str, Any]) -> RegistryPersistenceRequest:
    return RegistryPersistenceRequest(
        fedcore_id=fedcore_id,
        checkpoint_path=checkpoint_path,
        cleanup_after_save=cleanup_after_save,
        should_save_file=should_save_file,
        record=record,
    )


def execute_registry_persistence(request: RegistryPersistenceRequest,
                                 serialize_checkpoint: Callable[[], Optional[bytes]],
                                 save_checkpoint: Callable[..., None],
                                 append_record: Callable[[str, Dict[str, Any]], None]) -> None:
    if request.should_save_file:
        checkpoint_bytes = serialize_checkpoint()
        save_checkpoint(
            checkpoint_bytes,
            request.checkpoint_path,
            cleanup_after_save=request.cleanup_after_save,
        )

    append_record(request.fedcore_id, request.record)
