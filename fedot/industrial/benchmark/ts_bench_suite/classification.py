from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    ArtifactRecord,
    BenchmarkAggregateReport,
    BenchmarkRunRecord,
    BenchmarkSuiteConfig,
    ClassificationBenchmarkResult,
    ClassificationDatasetRecord,
    DatasetSpec,
    LabelPredictionRecord,
    MetricRecord,
    ModelSpec,
    RunStatus,
    TaskType,
    ensure_directory,
    new_run_id,
    to_plain_data,
    write_json,
)
from .local_io import LocalDatasetParseError, load_local_supervised_split
from .markdown import dataframe_to_markdown
from .progress import BenchmarkProgressMonitor

SUPPORTED_CLASSIFICATION_METRICS = ('accuracy', 'balanced_accuracy', 'f1_macro')


class BenchmarkClassificationError(ValueError):
    pass


def validate_tsc_suite_config(config: BenchmarkSuiteConfig) -> None:
    if config.task_type is not TaskType.TS_CLASSIFICATION:
        raise BenchmarkClassificationError('TSC suite expects task_type=ts_classification.')
    unsupported = set(config.metrics) - set(SUPPORTED_CLASSIFICATION_METRICS)
    if unsupported:
        raise BenchmarkClassificationError(f'Unsupported classification metrics: {sorted(unsupported)}')


def _normalize_matrix(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim > 2:
        return array.reshape(array.shape[0], -1)
    return array


def _normalize_labels(values: Any) -> np.ndarray:
    return np.asarray(values).reshape(-1).astype(str)


def _encode_dataset_record(
        benchmark: str,
        dataset_name: str,
        subset: str,
        train_features: Any,
        train_target: Any,
        test_features: Any,
        test_target: Any,
        metadata: dict[str, Any] | None = None,
) -> ClassificationDatasetRecord:
    train_x = _normalize_matrix(train_features)
    test_x = _normalize_matrix(test_features)
    train_y = _normalize_labels(train_target)
    test_y = _normalize_labels(test_target)
    return ClassificationDatasetRecord(
        benchmark=benchmark,
        dataset_name=dataset_name,
        subset=subset,
        train_features=tuple(tuple(float(value) for value in row) for row in train_x),
        train_target=tuple(str(value) for value in train_y),
        test_features=tuple(tuple(float(value) for value in row) for row in test_x),
        test_target=tuple(str(value) for value in test_y),
        metadata=metadata or {},
    )


class InMemoryClassificationAdapter:
    benchmark_name = 'in_memory_tsc'

    def load_dataset(self, spec: DatasetSpec) -> tuple[ClassificationDatasetRecord, ...]:
        payload = spec.adapter_options.get('record')
        if payload is None:
            raise BenchmarkClassificationError('In-memory TSC adapter expects adapter_options["record"].')
        return (
            _encode_dataset_record(
                benchmark=self.benchmark_name,
                dataset_name=spec.dataset_name,
                subset=spec.subset,
                train_features=payload['train_features'],
                train_target=payload['train_target'],
                test_features=payload['test_features'],
                test_target=payload['test_target'],
                metadata={'split_provenance': 'adapter_provided'},
            ),
        )


class LocalClassificationAdapter:
    benchmark_name = 'ucr_uea'

    def load_dataset(self, spec: DatasetSpec) -> tuple[ClassificationDatasetRecord, ...]:
        options = spec.adapter_options
        try:
            split = load_local_supervised_split(
                spec.dataset_name,
                data_root=options.get('local_data_root'),
                train_path=options.get('train_path'),
                test_path=options.get('test_path'),
            )
            train_x = split.train_features
            train_y = split.train_target
            test_x = split.test_features
            test_y = split.test_target
            metadata = dict(split.metadata)
        except LocalDatasetParseError:
            try:
                from fedot_ind.tools.loader import DataLoader
            except Exception as exc:  # pragma: no cover
                raise BenchmarkClassificationError(f'Classification loader is unavailable: {exc}') from exc
            train_data, test_data = DataLoader(dataset_name=spec.dataset_name).load_data()
            train_x, train_y = train_data
            test_x, test_y = test_data
            metadata = {'split_provenance': 'fedot_ind.tools.loader'}
        return (
            _encode_dataset_record(
                benchmark=self.benchmark_name,
                dataset_name=spec.dataset_name,
                subset=spec.subset,
                train_features=train_x,
                train_target=train_y,
                test_features=test_x,
                test_target=test_y,
                metadata=metadata,
            ),
        )


def build_classification_dataset_adapter(spec: DatasetSpec):
    benchmark = spec.benchmark.lower()
    if benchmark == 'in_memory_tsc':
        return InMemoryClassificationAdapter()
    if benchmark in {'ucr', 'uea', 'ucr_uea'}:
        return LocalClassificationAdapter()
    raise BenchmarkClassificationError(f'Unsupported classification adapter: {spec.benchmark}')


def compute_classification_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual = _normalize_labels(y_true)
    predicted = _normalize_labels(y_pred)
    labels = sorted(set(actual) | set(predicted))

    if metric_name == 'accuracy':
        return float(np.mean(actual == predicted))

    recalls = []
    f1_scores = []
    for label in labels:
        true_positive = int(np.sum((actual == label) & (predicted == label)))
        false_positive = int(np.sum((actual != label) & (predicted == label)))
        false_negative = int(np.sum((actual == label) & (predicted != label)))
        label_support = int(np.sum(actual == label))
        recall = true_positive / label_support if label_support else 0.0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        recalls.append(recall)
        f1_scores.append(f1)

    if metric_name == 'balanced_accuracy':
        return float(np.mean(recalls)) if recalls else 0.0
    if metric_name == 'f1_macro':
        return float(np.mean(f1_scores)) if f1_scores else 0.0
    raise BenchmarkClassificationError(f'Unsupported classification metric: {metric_name}')


@dataclass
class MajorityClassClassifier:
    name: str = 'MajorityClass'
    tags: tuple[str, ...] = ('baseline', 'classification')
    optional: bool = False
    majority_label_: str = ''

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        del features
        values, counts = np.unique(target, return_counts=True)
        self.majority_label_ = str(values[np.argmax(counts)])

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.majority_label_, dtype=object)


@dataclass
class NearestCentroidClassifier:
    name: str = 'NearestCentroid'
    tags: tuple[str, ...] = ('baseline', 'classification')
    optional: bool = False
    centroids_: dict[str, np.ndarray] | None = None

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        self.centroids_ = {}
        for label in np.unique(target):
            self.centroids_[str(label)] = np.mean(features[target == label], axis=0)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise BenchmarkClassificationError('NearestCentroidClassifier must be fitted before prediction.')
        predictions = []
        for row in features:
            label = min(
                self.centroids_.items(),
                key=lambda item: float(np.linalg.norm(row - item[1])),
            )[0]
            predictions.append(label)
        return np.asarray(predictions, dtype=object)


@dataclass
class OptionalExternalClassifier:
    dependency_name: str
    name: str
    tags: tuple[str, ...] = ('baseline', 'classification', 'external')
    optional: bool = True

    def availability(self) -> tuple[RunStatus, str]:
        try:
            __import__(self.dependency_name)
            return RunStatus.SKIPPED, 'Adapter scaffold registered but training backend is not wired yet.'
        except Exception:
            return RunStatus.NOT_AVAILABLE, f'{self.dependency_name} is not installed.'


def build_classification_model(spec: ModelSpec):
    name = spec.adapter_name.lower()
    if name == 'majority_class':
        return MajorityClassClassifier(name=spec.display_name, tags=spec.tags or ('baseline', 'classification'))
    if name == 'nearest_centroid':
        return NearestCentroidClassifier(name=spec.display_name, tags=spec.tags or ('baseline', 'classification'))
    if name == 'fedot_industrial_classifier':
        return OptionalExternalClassifier(
            dependency_name='fedot',
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'classification', 'external'),
        )
    raise BenchmarkClassificationError(f'Unsupported classification model adapter: {spec.adapter_name}')


def _build_classification_leaderboard(
        run_records: tuple[BenchmarkRunRecord, ...],
        primary_metric: str,
) -> BenchmarkAggregateReport:
    successful = [record for record in run_records if record.status is RunStatus.SUCCESS]
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in successful:
        metric_value = record.metrics_summary.get(primary_metric)
        if metric_value is not None:
            grouped.setdefault((record.dataset_name, record.model_name), []).append(metric_value)
    leaderboard_rows = []
    for (dataset_name, model_name), values in grouped.items():
        leaderboard_rows.append(
            {
                'dataset_name': dataset_name,
                'model_name': model_name,
                primary_metric: float(np.mean(values)),
                'n_runs': len(values),
            }
        )
    leaderboard_rows = sorted(leaderboard_rows, key=lambda row: row[primary_metric], reverse=True)
    for rank, row in enumerate(leaderboard_rows, start=1):
        row['rank'] = rank
    status_counts: dict[str, int] = {}
    for record in run_records:
        status_counts[record.status.value] = status_counts.get(record.status.value, 0) + 1
    run_id = run_records[0].run_id if run_records else new_run_id('empty_tsc')
    return BenchmarkAggregateReport(
        run_id=run_id,
        task_type=TaskType.TS_CLASSIFICATION,
        primary_metric=primary_metric,
        leaderboard_rows=tuple(leaderboard_rows),
        status_counts=status_counts,
    )


def run_tsc_suite(config: BenchmarkSuiteConfig) -> ClassificationBenchmarkResult:
    validate_tsc_suite_config(config)
    run_id = new_run_id(config.run_spec.run_name)
    dataset_records: list[ClassificationDatasetRecord] = []
    run_records: list[BenchmarkRunRecord] = []
    prediction_records: list[LabelPredictionRecord] = []
    metric_records: list[MetricRecord] = []
    progress = BenchmarkProgressMonitor(
        enabled=config.run_spec.show_progress,
        task_type=config.task_type.value,
        run_name=config.run_spec.run_name,
        leave=config.run_spec.progress_leave,
        log_errors=config.run_spec.progress_log_errors,
        log_summaries=config.run_spec.progress_log_summaries,
    )

    try:
        for dataset_spec in config.datasets:
            adapter = build_classification_dataset_adapter(dataset_spec)
            records = adapter.load_dataset(dataset_spec)
            dataset_records.extend(records)
            progress.extend_total(len(records) * len(config.models))
            progress.dataset_loaded(dataset_spec.dataset_name, len(records))
            for model_spec in config.models:
                model = build_classification_model(model_spec)
                progress.model_started(dataset_spec.dataset_name, model.name)
                availability_status, availability_message = model.availability()
                if availability_status is not RunStatus.SUCCESS:
                    for record in records:
                        progress.item_started(record.dataset_name, model.name, record.dataset_name)
                        run_records.append(
                            BenchmarkRunRecord(
                                run_id=run_id,
                                benchmark=record.benchmark,
                                dataset_name=record.dataset_name,
                                subset=record.subset,
                                series_id=record.dataset_name,
                                model_name=model.name,
                                status=availability_status,
                                tags=model.tags,
                                message=availability_message,
                            )
                        )
                        progress.advance(availability_status.value, availability_message)
                    progress.model_finished()
                    continue

                for record in records:
                    progress.item_started(record.dataset_name, model.name, record.dataset_name)
                    try:
                        train_x = np.asarray(record.train_features, dtype=float)
                        train_y = np.asarray(record.train_target, dtype=object)
                        test_x = np.asarray(record.test_features, dtype=float)
                        test_y = np.asarray(record.test_target, dtype=object)
                        model.fit(train_x, train_y)
                        prediction = np.asarray(model.predict(test_x), dtype=object)
                        metrics_summary = {
                            metric_name: compute_classification_metric(metric_name, test_y, prediction)
                            for metric_name in config.metrics
                        }
                        run_records.append(
                            BenchmarkRunRecord(
                                run_id=run_id,
                                benchmark=record.benchmark,
                                dataset_name=record.dataset_name,
                                subset=record.subset,
                                series_id=record.dataset_name,
                                model_name=model.name,
                                status=RunStatus.SUCCESS,
                                tags=model.tags,
                                metrics_summary=metrics_summary,
                            )
                        )
                        for metric_name, metric_value in metrics_summary.items():
                            metric_records.append(
                                MetricRecord(
                                    run_id=run_id,
                                    benchmark=record.benchmark,
                                    dataset_name=record.dataset_name,
                                    subset=record.subset,
                                    series_id=record.dataset_name,
                                    model_name=model.name,
                                    metric_name=metric_name,
                                    metric_value=metric_value,
                                    status=RunStatus.SUCCESS,
                                )
                            )
                        for sample_index, (actual, predicted) in enumerate(zip(test_y, prediction), start=1):
                            prediction_records.append(
                                LabelPredictionRecord(
                                    run_id=run_id,
                                    benchmark=record.benchmark,
                                    dataset_name=record.dataset_name,
                                    subset=record.subset,
                                    model_name=model.name,
                                    sample_index=sample_index,
                                    y_true=str(actual),
                                    y_pred=str(predicted),
                                    status=RunStatus.SUCCESS,
                                )
                            )
                        progress.advance(RunStatus.SUCCESS.value)
                    except Exception as exc:
                        run_records.append(
                            BenchmarkRunRecord(
                                run_id=run_id,
                                benchmark=record.benchmark,
                                dataset_name=record.dataset_name,
                                subset=record.subset,
                                series_id=record.dataset_name,
                                model_name=model.name,
                                status=RunStatus.FAILED,
                                tags=model.tags,
                                message=str(exc),
                            )
                        )
                        progress.advance(RunStatus.FAILED.value, str(exc))
                progress.model_finished()
            progress.dataset_finished()
    finally:
        progress.close()

    aggregate_report = _build_classification_leaderboard(tuple(run_records), config.run_spec.primary_metric)
    return ClassificationBenchmarkResult(
        run_id=run_id,
        config=config,
        dataset_records=tuple(dataset_records),
        run_records=tuple(run_records),
        prediction_records=tuple(prediction_records),
        metric_records=tuple(metric_records),
        aggregate_report=aggregate_report,
    )


def _frame_from_predictions(result: ClassificationBenchmarkResult) -> pd.DataFrame:
    return pd.DataFrame([to_plain_data(record) for record in result.prediction_records])


def _frame_from_metrics(result: ClassificationBenchmarkResult) -> pd.DataFrame:
    return pd.DataFrame([to_plain_data(record) for record in result.metric_records])


def _frame_from_runs(result: ClassificationBenchmarkResult) -> pd.DataFrame:
    rows = []
    for record in result.run_records:
        row = {
            'run_id': record.run_id,
            'benchmark': record.benchmark,
            'dataset_name': record.dataset_name,
            'subset': record.subset,
            'model_name': record.model_name,
            'status': record.status.value,
        }
        row.update(record.metrics_summary)
        rows.append(row)
    return pd.DataFrame(rows)


def render_tsc_publication_pack(
        result: ClassificationBenchmarkResult,
        output_dir: str | Path,
) -> tuple[ArtifactRecord, ...]:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    target_dir = ensure_directory(output_dir)
    aggregate_dir = ensure_directory(target_dir / 'aggregate')
    manifest: list[ArtifactRecord] = []

    predictions = _frame_from_predictions(result)
    metrics = _frame_from_metrics(result)
    runs = _frame_from_runs(result)
    leaderboard = pd.DataFrame(list(result.aggregate_report.leaderboard_rows))

    for name, frame in (
            ('predictions', predictions),
            ('metrics', metrics),
            ('runs', runs),
            ('leaderboard', leaderboard),
    ):
        csv_path = aggregate_dir / f'{name}.csv'
        tex_path = aggregate_dir / f'{name}.tex'
        frame.to_csv(csv_path, index=False)
        tex_path.write_text(frame.to_latex(index=False), encoding='utf-8')
        manifest.append(ArtifactRecord(kind='table', path=str(csv_path), format='csv'))
        manifest.append(ArtifactRecord(kind='table', path=str(tex_path), format='tex'))

    metadata_path = aggregate_dir / 'run_metadata.json'
    write_json(
        metadata_path,
        {
            'run_id': result.run_id,
            'task_type': result.config.task_type.value,
            'status_counts': result.aggregate_report.status_counts,
            'dataset_specs': [to_plain_data(spec) for spec in result.config.datasets],
            'model_specs': [to_plain_data(spec) for spec in result.config.models],
        },
    )
    manifest.append(ArtifactRecord(kind='structured', path=str(metadata_path), format='json'))

    summary_path = aggregate_dir / 'summary.md'
    summary_path.write_text(
        '\n'.join(
            [
                f'# TSC Benchmark Summary: {result.run_id}',
                '',
                f'- Primary metric: `{result.aggregate_report.primary_metric}`',
                f'- Successful runs: `{result.aggregate_report.status_counts.get("success", 0)}`',
                '',
                dataframe_to_markdown(leaderboard, index=False) if not leaderboard.empty else 'No successful runs.',
            ]
        ),
        encoding='utf-8',
    )
    manifest.append(ArtifactRecord(kind='summary', path=str(summary_path), format='md'))

    if not predictions.empty:
        dataset_name = str(predictions['dataset_name'].iloc[0])
        label_order = sorted(set(predictions['y_true']) | set(predictions['y_pred']))
        matrix = pd.crosstab(
            predictions['y_true'],
            predictions['y_pred'],
            rownames=['true'],
            colnames=['predicted'],
            dropna=False,
        ).reindex(index=label_order, columns=label_order, fill_value=0)
        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(matrix.values, cmap='Blues')
        axis.set_xticks(range(len(label_order)))
        axis.set_xticklabels(label_order, rotation=25, ha='right')
        axis.set_yticks(range(len(label_order)))
        axis.set_yticklabels(label_order)
        axis.set_title(f'Confusion Matrix: {dataset_name}')
        figure.colorbar(image, ax=axis)
        for extension in ('png', 'svg'):
            path = aggregate_dir / f'{dataset_name}_confusion_matrix.{extension}'
            figure.savefig(path, dpi=200, bbox_inches='tight')
            manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(figure)

    return tuple(manifest)
