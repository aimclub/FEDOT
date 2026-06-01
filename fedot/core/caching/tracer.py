import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

from fedot.core.caching.index_db import CacheIndexDB
from fedot.core.caching.normalization import normalize_for_hash, stable_hash
from fedot.core.caching.tools import ensure_cache_dirs
from fedot.core.utils import CACHE_DIR


@dataclass
class TraceModelRef:
    step_order: int
    model_hash: str
    model_path: str
    step_name: Optional[str] = None
    method: Optional[str] = None
    features_idx: Optional[Any] = None


@dataclass
class TraceStage:
    stage: str # obligatory_preprocessing/optional_preprocessing
    operation_hash: str # plan_hash
    input_hash: str
    output_hash: str
    tensor_data_path: Optional[str]
    operation_path: Optional[str] = None
    models: list[TraceModelRef] = field(default_factory=list)


@dataclass
class TraceManifest:
    trace_id: str
    trace_hash: str
    raw_fingerprint: str
    final_output_hash: str
    created_at: str
    stages: list[TraceStage]


class TraceBuilder:
    """
    Collects preprocessing cache artifacts produced during a single fit run.

    The builder does not own saving TensorData, plans or models. It reads their
    indexed metadata after cache writes complete and stores a readable manifest.
    """

    def __init__(self, raw_fingerprint: str, index_db: Optional[CacheIndexDB] = None):
        self.trace_id = str(uuid.uuid4())
        self.raw_fingerprint = raw_fingerprint
        self.index_db = index_db or CacheIndexDB()
        self.stages: list[TraceStage] = []

    def add_stage(self, stage: str, input_hash: str, operation_hash: str) -> TraceStage:
        tensor_record = self.index_db.get_tensor_data(input_hash, operation_hash)
        if tensor_record is None:
            raise ValueError(
                f"Cannot trace {stage}: TensorData cache record was not found "
                f"for input_hash={input_hash}, operation_hash={operation_hash}."
            )

        plan_record = self.index_db.get_preprocessing_plan(operation_hash)
        model_records = self.index_db.get_preprocessing_models(input_hash, operation_hash)
        models = [
            TraceModelRef(
                step_order=record.step_order,
                model_hash=record.model_hash,
                model_path=str(record.path),
                step_name=record.step_name,
                method=record.method,
                features_idx=record.features_idx,
            )
            for record in model_records
        ]

        trace_stage = TraceStage(
            stage=stage,
            operation_hash=operation_hash,
            input_hash=input_hash,
            output_hash=tensor_record.output_hash,
            tensor_data_path=str(tensor_record.path),
            operation_path=None if plan_record is None else str(plan_record.path),
            models=models,
        )
        self.stages.append(trace_stage)
        return trace_stage

    def finalize(self, final_output_hash: Optional[str] = None) -> TraceManifest:
        if final_output_hash is None:
            final_output_hash = self.stages[-1].output_hash if self.stages else self.raw_fingerprint

        created_at = datetime.now(timezone.utc).isoformat()
        trace_payload = {
            "trace_id": self.trace_id,
            "raw_fingerprint": self.raw_fingerprint,
            "final_output_hash": final_output_hash,
            "created_at": created_at,
            "stages": [asdict(stage) for stage in self.stages],
        }
        trace_hash = stable_hash(trace_payload)

        return TraceManifest(
            trace_id=self.trace_id,
            trace_hash=trace_hash,
            raw_fingerprint=self.raw_fingerprint,
            final_output_hash=final_output_hash,
            created_at=created_at,
            stages=self.stages,
        )

    def save(self, final_output_hash: Optional[str] = None) -> Path:
        ensure_cache_dirs()
        manifest = self.finalize(final_output_hash)
        trace_dir = CACHE_DIR / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{self.trace_id}.json"

        with open(trace_path, "w", encoding="utf-8") as file:
            json.dump(
                normalize_for_hash(asdict(manifest)),
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

        return trace_path