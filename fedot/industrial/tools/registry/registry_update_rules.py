"""Effect-light update/reporting rules for registry shell orchestration."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class RegisterChangesPlan:
    should_register_new: bool
    should_save_changes: bool
    stage: Optional[str]
    mode: Optional[str]


@dataclass(frozen=True)
class EvaluatorMetricsUpdatePlan:
    should_update: bool
    metrics: Dict[str, Any]


def build_register_changes_plan(existing_record: Optional[dict],
                                stage: Optional[str],
                                mode: Optional[str]) -> RegisterChangesPlan:
    has_existing = existing_record is not None
    return RegisterChangesPlan(
        should_register_new=not has_existing,
        should_save_changes=has_existing,
        stage=stage,
        mode=mode,
    )


def build_evaluator_metrics_update_plan(metrics_df: pd.DataFrame) -> EvaluatorMetricsUpdatePlan:
    if metrics_df.empty or len(metrics_df) == 0:
        return EvaluatorMetricsUpdatePlan(should_update=False, metrics={})

    last_gen_metrics = metrics_df.iloc[-1].to_dict()
    last_gen_metrics.pop('generation', None)

    if not last_gen_metrics:
        return EvaluatorMetricsUpdatePlan(should_update=False, metrics={})

    return EvaluatorMetricsUpdatePlan(
        should_update=True,
        metrics=last_gen_metrics,
    )
