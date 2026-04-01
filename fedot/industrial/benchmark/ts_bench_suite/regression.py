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
    DatasetSpec,
    MetricRecord,
    ModelSpec,
    RegressionBenchmarkResult,
    RegressionDatasetRecord,
    RunStatus,
    TaskType,
    ValuePredictionRecord,
    ensure_directory,
    new_run_id,
    to_plain_data,
    write_json,
)
from .local_io import LocalDatasetParseError, load_local_supervised_split
from .markdown import dataframe_to_markdown
from .progress import BenchmarkProgressMonitor

SUPPORTED_REGRESSION_METRICS = ('rmse', 'mae', 'r2')


class BenchmarkRegressionError(ValueError):
    pass


def validate_tser_suite_config(config: BenchmarkSuiteConfig) -> None:
    if config.task_type is not TaskType.TS_REGRESSION:
        raise BenchmarkRegressionError('TSER suite expects task_type=ts_regression.')
    unsupported = set(config.metrics) - set(SUPPORTED_REGRESSION_METRICS)
    if unsupported:
        raise BenchmarkRegressionError(f'Unsupported regression metrics: {sorted(unsupported)}')


def _normalize_matrix(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim > 2:
        return array.reshape(array.shape[0], -1)
    return array


def _normalize_vector(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _encode_dataset_record(
        benchmark: str,
        dataset_name: str,
        subset: str,
        train_features: Any,
        train_target: Any,
        test_features: Any,
        test_target: Any,
        metadata: dict[str, Any] | None = None,
) -> RegressionDatasetRecord:
    train_x = _normalize_matrix(train_features)
    test_x = _normalize_matrix(test_features)
    train_y = _normalize_vector(train_target)
    test_y = _normalize_vector(test_target)
    return RegressionDatasetRecord(
        benchmark=benchmark,
        dataset_name=dataset_name,
        subset=subset,
        train_features=tuple(tuple(float(value) for value in row) for row in train_x),
        train_target=tuple(float(value) for value in train_y),
        test_features=tuple(tuple(float(value) for value in row) for row in test_x),
        test_target=tuple(float(value) for value in test_y),
        metadata=metadata or {},
    )


class InMemoryRegressionAdapter:
    benchmark_name = 'in_memory_tser'

    def load_dataset(self, spec: DatasetSpec) -> tuple[RegressionDatasetRecord, ...]:
        payload = spec.adapter_options.get('record')
        if payload is None:
            raise BenchmarkRegressionError('In-memory TSER adapter expects adapter_options["record"].')
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


class LocalRegressionAdapter:
    benchmark_name = 'local_tser'

    def load_dataset(self, spec: DatasetSpec) -> tuple[RegressionDatasetRecord, ...]:
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
                raise BenchmarkRegressionError(f'Regression loader is unavailable: {exc}') from exc
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


def build_regression_dataset_adapter(spec: DatasetSpec):
    benchmark = spec.benchmark.lower()
    if benchmark == 'in_memory_tser':
        return InMemoryRegressionAdapter()
    if benchmark in {'local_tser', 'monash_regression'}:
        return LocalRegressionAdapter()
    raise BenchmarkRegressionError(f'Unsupported regression adapter: {spec.benchmark}')


def compute_regression_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual = _normalize_vector(y_true)
    predicted = _normalize_vector(y_pred)
    if metric_name == 'mae':
        return float(np.mean(np.abs(actual - predicted)))
    if metric_name == 'rmse':
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))
    if metric_name == 'r2':
        total = float(np.sum((actual - np.mean(actual)) ** 2))
        residual = float(np.sum((actual - predicted) ** 2))
        return float(1.0 - residual / total) if total > 1e-12 else 0.0
    raise BenchmarkRegressionError(f'Unsupported regression metric: {metric_name}')


@dataclass
class MeanRegressor:
    name: str = 'MeanRegressor'
    tags: tuple[str, ...] = ('baseline', 'regression')
    optional: bool = False
    mean_: float = 0.0

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        del features
        self.mean_ = float(np.mean(target))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.mean_, dtype=float)


@dataclass
class LinearRegressor:
    name: str = 'LinearRegressor'
    tags: tuple[str, ...] = ('baseline', 'regression')
    optional: bool = False
    coefficients_: np.ndarray | None = None

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        design = np.hstack([features, np.ones((features.shape[0], 1))])
        self.coefficients_, *_ = np.linalg.lstsq(design, target, rcond=None)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise BenchmarkRegressionError('LinearRegressor must be fitted before prediction.')
        design = np.hstack([features, np.ones((features.shape[0], 1))])
        return design @ self.coefficients_


@dataclass
class OptionalExternalRegressor:
    dependency_name: str
    name: str
    tags: tuple[str, ...] = ('baseline', 'regression', 'external')
    optional: bool = True

    def availability(self) -> tuple[RunStatus, str]:
        try:
            __import__(self.dependency_name)
            return RunStatus.SKIPPED, 'Adapter scaffold registered but training backend is not wired yet.'
        except Exception:
            return RunStatus.NOT_AVAILABLE, f'{self.dependency_name} is not installed.'


def build_regression_model(spec: ModelSpec):
    name = spec.adapter_name.lower()
    if name == 'mean_regressor':
        return MeanRegressor(name=spec.display_name, tags=spec.tags or ('baseline', 'regression'))
    if name == 'linear_regressor':
        return LinearRegressor(name=spec.display_name, tags=spec.tags or ('baseline', 'regression'))
    if name == 'fedot_industrial_regressor':
        return OptionalExternalRegressor(
            dependency_name='fedot',
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'regression', 'external'),
        )
    raise BenchmarkRegressionError(f'Unsupported regression model adapter: {spec.adapter_name}')


def _build_regression_leaderboard(
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
    reverse = primary_metric == 'r2'
    leaderboard_rows = sorted(leaderboard_rows, key=lambda row: row[primary_metric], reverse=reverse)
    for rank, row in enumerate(leaderboard_rows, start=1):
        row['rank'] = rank
    status_counts: dict[str, int] = {}
    for record in run_records:
        status_counts[record.status.value] = status_counts.get(record.status.value, 0) + 1
    run_id = run_records[0].run_id if run_records else new_run_id('empty_tser')
    return BenchmarkAggregateReport(
        run_id=run_id,
        task_type=TaskType.TS_REGRESSION,
        primary_metric=primary_metric,
        leaderboard_rows=tuple(leaderboard_rows),
        status_counts=status_counts,
    )


def run_tser_suite(config: BenchmarkSuiteConfig) -> RegressionBenchmarkResult:
    validate_tser_suite_config(config)
    run_id = new_run_id(config.run_spec.run_name)
    dataset_records: list[RegressionDatasetRecord] = []
    run_records: list[BenchmarkRunRecord] = []
    prediction_records: list[ValuePredictionRecord] = []
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
            adapter = build_regression_dataset_adapter(dataset_spec)
            records = adapter.load_dataset(dataset_spec)
            dataset_records.extend(records)
            progress.extend_total(len(records) * len(config.models))
            progress.dataset_loaded(dataset_spec.dataset_name, len(records))
            for model_spec in config.models:
                model = build_regression_model(model_spec)
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
                        train_y = np.asarray(record.train_target, dtype=float)
                        test_x = np.asarray(record.test_features, dtype=float)
                        test_y = np.asarray(record.test_target, dtype=float)
                        model.fit(train_x, train_y)
                        prediction = np.asarray(model.predict(test_x), dtype=float).reshape(-1)
                        metrics_summary = {
                            metric_name: compute_regression_metric(metric_name, test_y, prediction)
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
                                ValuePredictionRecord(
                                    run_id=run_id,
                                    benchmark=record.benchmark,
                                    dataset_name=record.dataset_name,
                                    subset=record.subset,
                                    model_name=model.name,
                                    sample_index=sample_index,
                                    y_true=float(actual),
                                    y_pred=float(predicted),
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

    aggregate_report = _build_regression_leaderboard(tuple(run_records), config.run_spec.primary_metric)
    return RegressionBenchmarkResult(
        run_id=run_id,
        config=config,
        dataset_records=tuple(dataset_records),
        run_records=tuple(run_records),
        prediction_records=tuple(prediction_records),
        metric_records=tuple(metric_records),
        aggregate_report=aggregate_report,
    )


def _frame_from_predictions(result: RegressionBenchmarkResult) -> pd.DataFrame:
    return pd.DataFrame([to_plain_data(record) for record in result.prediction_records])


def _frame_from_metrics(result: RegressionBenchmarkResult) -> pd.DataFrame:
    return pd.DataFrame([to_plain_data(record) for record in result.metric_records])


def _frame_from_runs(result: RegressionBenchmarkResult) -> pd.DataFrame:
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


def render_tser_publication_pack(
        result: RegressionBenchmarkResult,
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
                f'# TSER Benchmark Summary: {result.run_id}',
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
        scatter_figure, scatter_axis = plt.subplots(figsize=(6, 5))
        scatter_axis.scatter(predictions['y_true'], predictions['y_pred'], alpha=0.8)
        min_value = min(predictions['y_true'].min(), predictions['y_pred'].min())
        max_value = max(predictions['y_true'].max(), predictions['y_pred'].max())
        scatter_axis.plot([min_value, max_value], [min_value, max_value], linestyle='--', color='black')
        scatter_axis.set_title(f'Prediction vs Target: {dataset_name}')
        scatter_axis.set_xlabel('Target')
        scatter_axis.set_ylabel('Prediction')
        for extension in ('png', 'svg'):
            path = aggregate_dir / f'{dataset_name}_prediction_vs_target.{extension}'
            scatter_figure.savefig(path, dpi=200, bbox_inches='tight')
            manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(scatter_figure)

        residual_figure, residual_axis = plt.subplots(figsize=(6, 5))
        residuals = predictions['y_true'] - predictions['y_pred']
        residual_axis.scatter(predictions['y_pred'], residuals, alpha=0.8)
        residual_axis.axhline(0.0, linestyle='--', color='black')
        residual_axis.set_title(f'Residual Plot: {dataset_name}')
        residual_axis.set_xlabel('Prediction')
        residual_axis.set_ylabel('Residual')
        for extension in ('png', 'svg'):
            path = aggregate_dir / f'{dataset_name}_residuals.{extension}'
            residual_figure.savefig(path, dpi=200, bbox_inches='tight')
            manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(residual_figure)

    return tuple(manifest)
