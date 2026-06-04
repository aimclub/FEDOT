import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from fedot.core.caching.index_db import CacheIndexDB
from fedot.core.caching.normalization import normalize_for_hash, stable_hash
from fedot.core.caching.tools import ensure_cache_dirs
from fedot.core.utils import CACHE_DIR


@dataclass
class TraceModelRef:
    """Reference to one cached preprocessing model inside a trace stage."""
    step_order: int
    model_hash: str
    model_path: str
    step_name: Optional[str] = None
    method: Optional[str] = None
    features_idx: Optional[Any] = None


@dataclass
class TraceStage:
    """
    One preprocessing stage recorded in a trace manifest.

    Attributes:
        stage: Stage label such as ``obligatory_preprocessing``.
        operation_hash: Hash of the preprocessing plan applied at this stage.
        input_hash: Fingerprint of the stage input tensor data.
        output_hash: Fingerprint of the stage output tensor data.
        tensor_data_path: On-disk path to cached output ``TensorData``.
        operation_path: On-disk path to the cached preprocessing plan.
        models: Fitted preprocessing models used within the stage.
    """
    stage: str # obligatory_preprocessing/optional_preprocessing
    operation_hash: str # plan_hash
    input_hash: str
    output_hash: str
    tensor_data_path: Optional[str]
    operation_path: Optional[str] = None
    models: list[TraceModelRef] = field(default_factory=list)


@dataclass
class TraceManifest:
    """Serializable manifest describing one end-to-end preprocessing trace."""
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

    def __init__(
        self,
        raw_fingerprint: str,
        index_db: Optional[CacheIndexDB] = None,
        trace_id: Optional[str] = None,
        stages: Optional[list[TraceStage]] = None,
        created_at: Optional[str] = None,
    ):
        """
        Args:
            raw_fingerprint: Fingerprint of the raw input at the first stage.
            index_db: SQLite index used to resolve cached artifact paths.
            trace_id: Existing trace UUID or a newly generated value.
            stages: Previously recorded stages when reloading a manifest.
            created_at: ISO timestamp stored in the manifest.
        """
        self.trace_id = trace_id or str(uuid.uuid4())
        self.raw_fingerprint = raw_fingerprint
        self.index_db = index_db or CacheIndexDB()
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.stages: list[TraceStage] = stages or []

    @classmethod
    def from_trace_uuid(
        cls,
        trace_uuid: str,
        index_db: Optional[CacheIndexDB] = None,
    ) -> "TraceBuilder":
        """
        Restore a builder from an on-disk trace manifest.

        Args:
            trace_uuid: Identifier of the trace JSON file in ``CACHE_DIR/traces``.
            index_db: SQLite index used to resolve artifact metadata.

        Returns:
            Builder primed with stages loaded from the manifest.
        """
        trace_path = cls._trace_path(trace_uuid)
        with open(trace_path, encoding="utf-8") as file:
            manifest = json.load(file)

        stages = [
            cls._stage_from_dict(stage_data)
            for stage_data in manifest.get("stages", [])
        ]
        return cls(
            raw_fingerprint=manifest["raw_fingerprint"],
            index_db=index_db,
            trace_id=manifest["trace_id"],
            stages=stages,
            created_at=manifest.get("created_at"),
        )

    def add_stage(self, stage: str, input_hash: str, operation_hash: str) -> TraceStage:
        """
        Append a preprocessing stage using metadata already stored in the index.

        Args:
            stage: Stage label written to the manifest.
            input_hash: Fingerprint of the stage input.
            operation_hash: Fingerprint of the preprocessing plan.

        Returns:
            Newly created ``TraceStage`` instance.

        Raises:
            ValueError: When the tensor-data index row is missing.
        """
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
            tensor_data_path=None if tensor_record.path is None else str(tensor_record.path),
            operation_path=None if plan_record is None else str(plan_record.path),
            models=models,
        )
        self.stages.append(trace_stage)
        return trace_stage

    def finalize(self, final_output_hash: Optional[str] = None) -> TraceManifest:
        """
        Build an in-memory manifest and compute its content hash.

        Args:
            final_output_hash: Output fingerprint of the last stage. When omitted,
                the last stage output or ``raw_fingerprint`` is used.

        Returns:
            ``TraceManifest`` ready for JSON serialization.
        """
        if final_output_hash is None:
            final_output_hash = self.stages[-1].output_hash if self.stages else self.raw_fingerprint

        trace_payload = {
            "trace_id": self.trace_id,
            "raw_fingerprint": self.raw_fingerprint,
            "final_output_hash": final_output_hash,
            "created_at": self.created_at,
            "stages": [asdict(stage) for stage in self.stages],
        }
        trace_hash = stable_hash(trace_payload)

        return TraceManifest(
            trace_id=self.trace_id,
            trace_hash=trace_hash,
            raw_fingerprint=self.raw_fingerprint,
            final_output_hash=final_output_hash,
            created_at=self.created_at,
            stages=self.stages,
        )

    def save(self, final_output_hash: Optional[str] = None) -> Path:
        """
        Write the trace manifest to ``CACHE_DIR/traces/{trace_id}.json``.

        Args:
            final_output_hash: Output fingerprint stored in the manifest.

        Returns:
            Path to the saved JSON file.
        """
        ensure_cache_dirs()
        manifest = self.finalize(final_output_hash)
        trace_path = self._trace_path(self.trace_id)
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        with open(trace_path, "w", encoding="utf-8") as file:
            json.dump(
                normalize_for_hash(asdict(manifest)),
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

        return trace_path
    
    @staticmethod
    def update_according_to_cache(tensor_data_hashes: List[str]) -> None:
        """
        Sync trace manifests after TensorData cache entries were removed.

        For each stage whose ``output_hash`` was cleared, drop ``tensor_data_path``
        when the referenced file is missing and refresh ``trace_hash``.
        """
        if not tensor_data_hashes:
            return

        ensure_cache_dirs()
        cleared_hashes = set(tensor_data_hashes)
        traces_dir = CACHE_DIR / "traces"
        if not traces_dir.exists():
            return

        for trace_path in traces_dir.glob("*.json"):
            with open(trace_path, encoding="utf-8") as file:
                manifest = json.load(file)

            modified = False
            for stage in manifest.get("stages", []):
                if stage.get("output_hash") not in cleared_hashes:
                    continue

                tensor_data_path = stage.get("tensor_data_path")
                if tensor_data_path and Path(tensor_data_path).is_file():
                    continue

                if tensor_data_path != "":
                    stage["tensor_data_path"] = ""
                    modified = True

            if not modified:
                continue

            trace_payload = {
                "trace_id": manifest["trace_id"],
                "raw_fingerprint": manifest["raw_fingerprint"],
                "final_output_hash": manifest["final_output_hash"],
                "created_at": manifest["created_at"],
                "stages": manifest["stages"],
            }
            manifest["trace_hash"] = stable_hash(trace_payload)

            with open(trace_path, "w", encoding="utf-8") as file:
                json.dump(
                    normalize_for_hash(manifest),
                    file,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )

    @staticmethod
    def _trace_path(trace_uuid: str) -> Path:
        """Return the filesystem path of a trace manifest."""
        return CACHE_DIR / "traces" / f"{trace_uuid}.json"

    @staticmethod
    def _stage_from_dict(stage_data: dict[str, Any]) -> TraceStage:
        """Deserialize one trace stage from manifest JSON."""
        return TraceStage(
            stage=stage_data["stage"],
            operation_hash=stage_data["operation_hash"],
            input_hash=stage_data["input_hash"],
            output_hash=stage_data["output_hash"],
            tensor_data_path=stage_data.get("tensor_data_path"),
            operation_path=stage_data.get("operation_path"),
            models=[
                TraceModelRef(**model_data)
                for model_data in stage_data.get("models", [])
            ],
        )