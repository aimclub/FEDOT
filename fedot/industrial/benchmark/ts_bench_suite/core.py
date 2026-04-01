from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Sequence


class TaskType(str, Enum):
    FORECASTING = 'forecasting'
    TS_CLASSIFICATION = 'ts_classification'
    TS_REGRESSION = 'ts_regression'


class RunStatus(str, Enum):
    SUCCESS = 'success'
    FAILED = 'failed'
    SKIPPED = 'skipped'
    NOT_AVAILABLE = 'not_available'


@dataclass(frozen=True)
class DatasetSpec:
    benchmark: str
    dataset_name: str
    subset: str = 'default'
    sample_size: int | None = None
    random_seed: int = 0
    series_ids: tuple[str, ...] = ()
    adapter_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    adapter_name: str
    display_name: str
    tags: tuple[str, ...] = ()
    optional: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    run_name: str = 'benchmark_v2'
    random_seed: int = 0
    primary_metric: str = 'owa'
    show_progress: bool = True
    progress_leave: bool = True
    progress_log_errors: bool = True
    progress_log_summaries: bool = True


@dataclass(frozen=True)
class ArtifactSpec:
    output_dir: str
    plot_formats: tuple[str, ...] = ('png', 'svg')
    table_formats: tuple[str, ...] = ('csv', 'tex')
    structured_formats: tuple[str, ...] = ('json', 'parquet')
    summary_formats: tuple[str, ...] = ('md',)
    persist_on_run: bool = True
    render_publication_pack: bool = False


@dataclass(frozen=True)
class BenchmarkSuiteConfig:
    task_type: TaskType
    datasets: tuple[DatasetSpec, ...]
    models: tuple[ModelSpec, ...]
    artifact_spec: ArtifactSpec
    run_spec: RunSpec = field(default_factory=RunSpec)
    metrics: tuple[str, ...] = ('mase', 'smape', 'owa', 'rmse', 'mae')


@dataclass(frozen=True)
class ForecastingSeriesRecord:
    benchmark: str
    dataset_name: str
    subset: str
    series_id: str
    frequency: str
    forecast_horizon: int
    seasonal_period: int
    train_values: tuple[float, ...]
    test_values: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    benchmark: str
    dataset_name: str
    subset: str
    series_id: str
    model_name: str
    horizon_index: int
    y_true: float
    y_pred: float
    status: RunStatus


@dataclass(frozen=True)
class MetricRecord:
    run_id: str
    benchmark: str
    dataset_name: str
    subset: str
    series_id: str
    model_name: str
    metric_name: str
    metric_value: float
    status: RunStatus
    horizon_index: int | None = None


@dataclass(frozen=True)
class BenchmarkRunRecord:
    run_id: str
    benchmark: str
    dataset_name: str
    subset: str
    series_id: str
    model_name: str
    status: RunStatus
    tags: tuple[str, ...] = ()
    message: str = ''
    metrics_summary: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactRecord:
    kind: str
    path: str
    format: str


@dataclass(frozen=True)
class BenchmarkAggregateReport:
    run_id: str
    task_type: TaskType
    primary_metric: str
    leaderboard_rows: tuple[dict[str, Any], ...]
    status_counts: dict[str, int]


@dataclass(frozen=True)
class ForecastingBenchmarkResult:
    run_id: str
    config: BenchmarkSuiteConfig
    series_records: tuple[ForecastingSeriesRecord, ...]
    run_records: tuple[BenchmarkRunRecord, ...]
    prediction_records: tuple[PredictionRecord, ...]
    metric_records: tuple[MetricRecord, ...]
    aggregate_report: BenchmarkAggregateReport
    artifact_manifest: tuple[ArtifactRecord, ...] = ()


@dataclass(frozen=True)
class ClassificationDatasetRecord:
    benchmark: str
    dataset_name: str
    subset: str
    train_features: tuple[tuple[float, ...], ...]
    train_target: tuple[str, ...]
    test_features: tuple[tuple[float, ...], ...]
    test_target: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegressionDatasetRecord:
    benchmark: str
    dataset_name: str
    subset: str
    train_features: tuple[tuple[float, ...], ...]
    train_target: tuple[float, ...]
    test_features: tuple[tuple[float, ...], ...]
    test_target: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelPredictionRecord:
    run_id: str
    benchmark: str
    dataset_name: str
    subset: str
    model_name: str
    sample_index: int
    y_true: str
    y_pred: str
    status: RunStatus


@dataclass(frozen=True)
class ValuePredictionRecord:
    run_id: str
    benchmark: str
    dataset_name: str
    subset: str
    model_name: str
    sample_index: int
    y_true: float
    y_pred: float
    status: RunStatus


@dataclass(frozen=True)
class ClassificationBenchmarkResult:
    run_id: str
    config: BenchmarkSuiteConfig
    dataset_records: tuple[ClassificationDatasetRecord, ...]
    run_records: tuple[BenchmarkRunRecord, ...]
    prediction_records: tuple[LabelPredictionRecord, ...]
    metric_records: tuple[MetricRecord, ...]
    aggregate_report: BenchmarkAggregateReport
    artifact_manifest: tuple[ArtifactRecord, ...] = ()


@dataclass(frozen=True)
class RegressionBenchmarkResult:
    run_id: str
    config: BenchmarkSuiteConfig
    dataset_records: tuple[RegressionDatasetRecord, ...]
    run_records: tuple[BenchmarkRunRecord, ...]
    prediction_records: tuple[ValuePredictionRecord, ...]
    metric_records: tuple[MetricRecord, ...]
    aggregate_report: BenchmarkAggregateReport
    artifact_manifest: tuple[ArtifactRecord, ...] = ()


class ForecastingDatasetAdapter(Protocol):
    benchmark_name: str

    def load_series(self, spec: DatasetSpec) -> tuple[ForecastingSeriesRecord, ...]:
        ...


class ForecastingModelAdapter(Protocol):
    name: str
    tags: tuple[str, ...]
    optional: bool

    def availability(self) -> tuple[RunStatus, str]:
        ...

    def forecast(
            self,
            series_record: ForecastingSeriesRecord,
    ) -> tuple[Sequence[float], dict[str, Any]]:
        ...


def new_run_id(prefix: str = 'benchmark_v2') -> str:
    return f'{prefix}_{uuid.uuid4().hex[:10]}'


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist') and value.__class__.__module__.startswith('numpy'):
        return to_plain_data(value.tolist())
    if hasattr(value, 'item') and value.__class__.__module__.startswith('numpy'):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    return value


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(to_plain_data(payload), stream, indent=2, ensure_ascii=False)
