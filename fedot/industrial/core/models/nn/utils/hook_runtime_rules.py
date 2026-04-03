from __future__ import annotations

from typing import Any, Literal


HookStage = Literal['start', 'end']


def resolve_stage_hooks(hooks_collection: Any, stage: HookStage) -> list[Any]:
    if stage == 'start':
        return list(hooks_collection.start)
    if stage == 'end':
        return list(hooks_collection.end)
    raise ValueError(f'Unsupported hook stage: {stage}')


def build_hook_runtime_payload(*,
                               trainer_objects: dict[str, Any],
                               history: dict[str, Any],
                               learning_rate: float | None = None,
                               val_loader: Any = None,
                               criterion: Any = None,
                               extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        'trainer_objects': trainer_objects,
        'history': history,
    }
    if learning_rate is not None:
        payload['learning_rate'] = learning_rate
    if val_loader is not None:
        payload['val_loader'] = val_loader
    if criterion is not None:
        payload['criterion'] = criterion
    if extra:
        payload.update(extra)
    return payload


def execute_stage_hooks(hooks_collection: Any,
                        stage: HookStage,
                        epoch: int,
                        payload: dict[str, Any]) -> None:
    for hook in resolve_stage_hooks(hooks_collection, stage):
        hook(epoch=epoch, **payload)


def should_stop_training(trainer_objects: dict[str, Any]) -> bool:
    return bool(trainer_objects.get('stop', False))
