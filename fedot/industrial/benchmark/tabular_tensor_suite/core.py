from __future__ import annotations

import csv
import json
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


class RunStatus(str, Enum):
    SUCCESS = 'success'
    FAILED = 'failed'
    SKIPPED = 'skipped'


class ExecutorKind(str, Enum):
    SEQUENTIAL = 'sequential'
    DASK_EXPERIMENTAL = 'dask_experimental'


@dataclass(frozen=True)
class TabularDatasetSpec:
    name: str
    problem: str
    loader_kind: str
    target_column: str | None = None
    path: str | None = None
    test_size: float = 0.2
    loader_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedTabularDataset:
    spec: TabularDatasetSpec
    features: Any
    target: Any
    sample_count: int
    feature_count: int
    size_bucket: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorBenchmarkConfig:
    datasets: tuple[str, ...]
    modes: tuple[str, ...]
    executor: ExecutorKind
    seeds: tuple[int, ...]
    generations: int = 1
    pop_size: int = 15
    output_dir: str = 'benchmark_results/tabular_tensor_suite'
    timeout_minutes: float = 5.0
    cv_folds: int | None = None


@dataclass(frozen=True)
class SuiteCapabilities:
    gpu_available: bool
    dask_available: bool
    visible_gpu_count: int
    fedot_runtime_available: bool = True
    gpu_reason: str = ''
    dask_reason: str = ''
    fedot_runtime_reason: str = ''


@dataclass(frozen=True)
class IndividualBenchmarkRecord:
    dataset_name: str
    problem: str
    size_bucket: str
    mode: str
    seed: int
    executor: str
    pipeline_id: str
    n_nodes: int
    n_model_nodes: int
    fit_time_sec: float
    predict_time_sec: float
    objective_value: float | None
    success: bool
    failure_reason: str
    device: str
    data_mode: str
    cpu_mem_mb: float | None = None
    gpu_mem_mb_peak: float | None = None


@dataclass(frozen=True)
class GenerationBenchmarkRecord:
    dataset_name: str
    problem: str
    size_bucket: str
    mode: str
    seed: int
    executor: str
    status: str
    skip_reason: str
    device: str
    data_mode: str
    quality_metric_name: str
    quality_metric_value: float | None
    wall_clock_sec: float | None
    mean_fit_sec: float | None
    median_fit_sec: float | None
    p95_fit_sec: float | None
    success_rate: float | None
    throughput_individuals_per_min: float | None
    best_metric: float | None
    top3_mean_metric: float | None
    unique_pipeline_ratio: float | None


@dataclass(frozen=True)
class TensorBenchmarkSuiteResult:
    run_id: str
    config: TensorBenchmarkConfig
    capabilities: SuiteCapabilities
    individual_records: tuple[IndividualBenchmarkRecord, ...]
    generation_records: tuple[GenerationBenchmarkRecord, ...]
    artifact_paths: tuple[str, ...] = ()


def new_run_id(prefix: str = 'tensor_suite') -> str:
    return f'{prefix}_{uuid.uuid4().hex[:10]}'


def resolve_size_bucket(sample_count: int) -> str:
    if sample_count < 1000:
        return 'small'
    if sample_count <= 10000:
        return 'medium'
    return 'large'


def parse_csv_tuple(raw_value: str | None, cast=str) -> tuple[Any, ...]:
    if raw_value is None:
        return tuple()
    values = []
    for raw_item in raw_value.split(','):
        normalized = raw_item.strip()
        if not normalized:
            continue
        values.append(cast(normalized))
    return tuple(values)


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain_data(field_value) for key, field_value in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_plain_data(field_value) for key, field_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: str | Path, payload: Any) -> None:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(to_plain_data(payload), indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


def write_csv_records(path: str | Path, records: Sequence[Any]) -> None:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [to_plain_data(record) for record in records]
    fieldnames = sorted({key for row in rows for key in row.keys()})

    with resolved_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_output_dir(path: str | Path) -> Path:
    resolved_path = Path(path)
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def safe_mean(values: Iterable[float]) -> float | None:
    normalized = tuple(values)
    if not normalized:
        return None
    return float(np.mean(normalized))


def safe_percentile(values: Iterable[float], percentile: float) -> float | None:
    normalized = tuple(values)
    if not normalized:
        return None
    return float(np.percentile(normalized, percentile))
