"""Pure rules for registry stage, mode, and record planning."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Literal

_BEFORE_STAGE_KEYWORDS = ('before', 'initial')

RegistryModeSource = Literal['explicit', 'inherited', 'trainer', 'none']


@dataclass(frozen=True)
class RegistryStageModePlan:
    stage: Optional[str]
    mode: Optional[str]
    mode_source: RegistryModeSource


@dataclass(frozen=True)
class RegistryRecordPlan:
    record: str
    fedcore: str
    model: str
    created_at: str
    model_path: Optional[str]
    checkpoint_path: str
    stage: Optional[str]
    mode: Optional[str]
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistryRecordUpdatePlan:
    stage: Optional[str]
    mode: Optional[str]
    metrics: Any


def normalize_registry_stage(stage: Optional[str]) -> Optional[str]:
    if stage is None:
        return None

    stage_lower = str(stage).lower()
    if any(keyword in stage_lower for keyword in _BEFORE_STAGE_KEYWORDS):
        return 'before'
    return 'after'


def resolve_registry_mode(mode: Optional[str],
                          latest_record: Optional[Mapping[str, Any]] = None,
                          trainer: Any = None) -> RegistryStageModePlan:
    if mode is not None:
        return RegistryStageModePlan(stage=None, mode=mode, mode_source='explicit')

    if latest_record is not None:
        inherited_mode = latest_record.get('mode')
        if inherited_mode:
            return RegistryStageModePlan(stage=None, mode=inherited_mode, mode_source='inherited')

    trainer_name = getattr(getattr(trainer, '__class__', None), '__name__', None)
    if trainer_name:
        return RegistryStageModePlan(stage=None, mode=trainer_name, mode_source='trainer')

    return RegistryStageModePlan(stage=None, mode=None, mode_source='none')


def build_registry_stage_mode_plan(stage: Optional[str],
                                   mode: Optional[str],
                                   latest_record: Optional[Mapping[str, Any]] = None,
                                   trainer: Any = None) -> RegistryStageModePlan:
    mode_plan = resolve_registry_mode(mode, latest_record=latest_record, trainer=trainer)
    return RegistryStageModePlan(
        stage=normalize_registry_stage(stage),
        mode=mode_plan.mode,
        mode_source=mode_plan.mode_source,
    )


def build_registry_record_plan(record_id: str,
                               fedcore_id: str,
                               model_id: str,
                               version: str,
                               checkpoint_path: str,
                               model_path: Optional[str] = None,
                               stage: Optional[str] = None,
                               mode: Optional[str] = None) -> RegistryRecordPlan:
    return RegistryRecordPlan(
        record=str(record_id),
        fedcore=fedcore_id,
        model=model_id,
        created_at=version,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        stage=stage,
        mode=mode,
        metrics={},
    )


def merge_registry_metrics(current_metrics: Any, new_metrics: Any) -> Any:
    if not isinstance(new_metrics, dict):
        return new_metrics

    base_metrics = current_metrics if isinstance(current_metrics, dict) else {}
    return {**base_metrics, **new_metrics}


def build_registry_record_update_plan(current_metrics: Any,
                                      new_metrics: Any,
                                      stage: Optional[str] = None,
                                      mode: Optional[str] = None,
                                      trainer: Any = None) -> RegistryRecordUpdatePlan:
    stage_mode_plan = build_registry_stage_mode_plan(stage, mode, trainer=trainer)
    return RegistryRecordUpdatePlan(
        stage=stage_mode_plan.stage,
        mode=stage_mode_plan.mode,
        metrics=merge_registry_metrics(current_metrics, new_metrics),
    )
