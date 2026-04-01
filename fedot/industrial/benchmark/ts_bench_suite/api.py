from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from .analytics import compare_models_on_series, render_publication_pack
from .core import (
    ArtifactRecord,
    ArtifactSpec,
    BenchmarkSuiteConfig,
    ClassificationBenchmarkResult,
    DatasetSpec,
    ForecastingBenchmarkResult,
    ModelSpec,
    RegressionBenchmarkResult,
    RunSpec,
    TaskType,
    write_json,
)
from .classification import render_tsc_publication_pack, run_tsc_suite
from .forecasting import run_forecasting_suite
from .regression import render_tser_publication_pack, run_tser_suite


def _build_issue_artifacts(result, output_dir: str | Path) -> tuple[ArtifactRecord, ...]:
    target_dir = Path(output_dir)
    issues = []
    for record in result.run_records:
        if record.status.value == 'success':
            continue
        issues.append(
            {
                'run_id': record.run_id,
                'benchmark': record.benchmark,
                'dataset_name': record.dataset_name,
                'subset': record.subset,
                'series_id': record.series_id,
                'model_name': record.model_name,
                'status': record.status.value,
                'message': record.message,
                'tags': list(record.tags),
                'metadata': record.metadata,
            }
        )
    if not issues:
        return ()

    jsonl_path = target_dir / 'errors.jsonl'
    with jsonl_path.open('w', encoding='utf-8') as stream:
        for issue in issues:
            stream.write(json.dumps(issue, ensure_ascii=False) + '\n')

    summary_path = target_dir / 'errors_summary.json'
    status_counts: dict[str, int] = {}
    for issue in issues:
        status = str(issue['status'])
        status_counts[status] = status_counts.get(status, 0) + 1
    write_json(
        summary_path,
        {
            'run_id': result.run_id,
            'issue_count': len(issues),
            'status_counts': status_counts,
        },
    )
    return (
        ArtifactRecord(kind='structured', path=str(jsonl_path), format='jsonl'),
        ArtifactRecord(kind='structured', path=str(summary_path), format='json'),
    )


def run_forecasting_benchmark_suite(config: BenchmarkSuiteConfig) -> ForecastingBenchmarkResult:
    result = run_forecasting_suite(config)
    if config.artifact_spec.persist_on_run:
        output_dir = Path(config.artifact_spec.output_dir) / result.run_id
        manifest = list(
            render_publication_pack(
                result,
                output_dir=output_dir,
            )
        )
        manifest.extend(_build_issue_artifacts(result, output_dir))
        result = ForecastingBenchmarkResult(
            run_id=result.run_id,
            config=result.config,
            series_records=result.series_records,
            run_records=result.run_records,
            prediction_records=result.prediction_records,
            metric_records=result.metric_records,
            aggregate_report=result.aggregate_report,
            artifact_manifest=tuple(manifest),
        )
    return result


def compare_forecasting_models_on_series(
        result: ForecastingBenchmarkResult,
        series_id: str,
        output_dir: str | Path | None = None,
):
    return compare_models_on_series(result, series_id=series_id, output_dir=output_dir)


def build_forecasting_publication_pack(
        result: ForecastingBenchmarkResult,
        output_dir: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    return render_publication_pack(result, output_dir=output_dir)


def build_tsc_publication_pack(
        result: ClassificationBenchmarkResult,
        output_dir: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    return render_tsc_publication_pack(result, output_dir=output_dir)


def build_tser_publication_pack(
        result: RegressionBenchmarkResult,
        output_dir: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    return render_tser_publication_pack(result, output_dir=output_dir)


def run_tsc_benchmark_suite(config: BenchmarkSuiteConfig):
    if config.task_type is not TaskType.TS_CLASSIFICATION:
        raise ValueError('run_tsc_benchmark_suite expects task_type=ts_classification.')
    result = run_tsc_suite(config)
    if config.artifact_spec.persist_on_run:
        output_dir = Path(config.artifact_spec.output_dir) / result.run_id
        manifest = list(
            render_tsc_publication_pack(
                result,
                output_dir=output_dir,
            )
        )
        manifest.extend(_build_issue_artifacts(result, output_dir))
        result = ClassificationBenchmarkResult(
            run_id=result.run_id,
            config=result.config,
            dataset_records=result.dataset_records,
            run_records=result.run_records,
            prediction_records=result.prediction_records,
            metric_records=result.metric_records,
            aggregate_report=result.aggregate_report,
            artifact_manifest=tuple(manifest),
        )
    return result


def run_tser_benchmark_suite(config: BenchmarkSuiteConfig):
    if config.task_type is not TaskType.TS_REGRESSION:
        raise ValueError('run_tser_benchmark_suite expects task_type=ts_regression.')
    result = run_tser_suite(config)
    if config.artifact_spec.persist_on_run:
        output_dir = Path(config.artifact_spec.output_dir) / result.run_id
        manifest = list(
            render_tser_publication_pack(
                result,
                output_dir=output_dir,
            )
        )
        manifest.extend(_build_issue_artifacts(result, output_dir))
        result = RegressionBenchmarkResult(
            run_id=result.run_id,
            config=result.config,
            dataset_records=result.dataset_records,
            run_records=result.run_records,
            prediction_records=result.prediction_records,
            metric_records=result.metric_records,
            aggregate_report=result.aggregate_report,
            artifact_manifest=tuple(manifest),
        )
    return result


def build_legacy_tsf_suite_config(experiment_setup: dict[str, Any]) -> BenchmarkSuiteConfig:
    dataset_payloads = tuple(experiment_setup.get('dataset_specs', ()))
    model_payloads = tuple(experiment_setup.get('model_specs', ()))
    output_dir = experiment_setup.get('output_dir', './benchmark/results/v2')
    metrics = tuple(experiment_setup.get('metrics', ('mase', 'smape', 'owa', 'rmse', 'mae')))

    if not dataset_payloads or not model_payloads:
        raise ValueError('Legacy v2 compatibility expects experiment_setup["dataset_specs"] and ["model_specs"].')

    datasets = tuple(DatasetSpec(**payload) for payload in dataset_payloads)
    models = tuple(ModelSpec(**payload) for payload in model_payloads)
    artifact_spec = experiment_setup.get('artifact_spec')
    if artifact_spec is None:
        artifact_spec = ArtifactSpec(output_dir=output_dir)
    run_spec = experiment_setup.get('run_spec')
    if run_spec is None:
        run_spec = RunSpec(run_name='legacy_benchmark_tsf')

    return BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=datasets,
        models=models,
        metrics=metrics,
        artifact_spec=artifact_spec,
        run_spec=run_spec,
    )


def run_forecasting_benchmark_from_legacy_config(experiment_setup: dict[str, Any]) -> ForecastingBenchmarkResult:
    warnings.warn(
        'benchmark.BenchmarkTSF legacy path is delegating to benchmark.v2 forecasting suite.',
        DeprecationWarning,
        stacklevel=2,
    )
    config = build_legacy_tsf_suite_config(experiment_setup)
    return run_forecasting_benchmark_suite(config)


def build_legacy_tsc_suite_config(experiment_setup: dict[str, Any]) -> BenchmarkSuiteConfig:
    dataset_payloads = tuple(experiment_setup.get('dataset_specs', ()))
    model_payloads = tuple(experiment_setup.get('model_specs', ()))
    output_dir = experiment_setup.get('output_dir', './benchmark/results/v2_tsc')
    metrics = tuple(experiment_setup.get('metrics', ('accuracy', 'balanced_accuracy', 'f1_macro')))
    if not dataset_payloads or not model_payloads:
        raise ValueError('Legacy TSC v2 compatibility expects experiment_setup["dataset_specs"] and ["model_specs"].')
    datasets = tuple(DatasetSpec(**payload) for payload in dataset_payloads)
    models = tuple(ModelSpec(**payload) for payload in model_payloads)
    artifact_spec = experiment_setup.get('artifact_spec') or ArtifactSpec(output_dir=output_dir)
    run_spec = experiment_setup.get('run_spec') or RunSpec(run_name='legacy_benchmark_tsc', primary_metric=metrics[0])
    return BenchmarkSuiteConfig(
        task_type=TaskType.TS_CLASSIFICATION,
        datasets=datasets,
        models=models,
        metrics=metrics,
        artifact_spec=artifact_spec,
        run_spec=run_spec,
    )


def build_legacy_tser_suite_config(experiment_setup: dict[str, Any]) -> BenchmarkSuiteConfig:
    dataset_payloads = tuple(experiment_setup.get('dataset_specs', ()))
    model_payloads = tuple(experiment_setup.get('model_specs', ()))
    output_dir = experiment_setup.get('output_dir', './benchmark/results/v2_tser')
    metrics = tuple(experiment_setup.get('metrics', ('rmse', 'mae', 'r2')))
    if not dataset_payloads or not model_payloads:
        raise ValueError('Legacy TSER v2 compatibility expects experiment_setup["dataset_specs"] and ["model_specs"].')
    datasets = tuple(DatasetSpec(**payload) for payload in dataset_payloads)
    models = tuple(ModelSpec(**payload) for payload in model_payloads)
    artifact_spec = experiment_setup.get('artifact_spec') or ArtifactSpec(output_dir=output_dir)
    run_spec = experiment_setup.get('run_spec') or RunSpec(run_name='legacy_benchmark_tser', primary_metric=metrics[0])
    return BenchmarkSuiteConfig(
        task_type=TaskType.TS_REGRESSION,
        datasets=datasets,
        models=models,
        metrics=metrics,
        artifact_spec=artifact_spec,
        run_spec=run_spec,
    )


def run_tsc_benchmark_from_legacy_config(experiment_setup: dict[str, Any]):
    warnings.warn(
        'benchmark.BenchmarkTSC legacy path is delegating to benchmark.v2 classification suite.',
        DeprecationWarning,
        stacklevel=2,
    )
    config = build_legacy_tsc_suite_config(experiment_setup)
    return run_tsc_benchmark_suite(config)


def run_tser_benchmark_from_legacy_config(experiment_setup: dict[str, Any]):
    warnings.warn(
        'benchmark.BenchmarkTSER legacy path is delegating to benchmark.v2 regression suite.',
        DeprecationWarning,
        stacklevel=2,
    )
    config = build_legacy_tser_suite_config(experiment_setup)
    return run_tser_benchmark_suite(config)
