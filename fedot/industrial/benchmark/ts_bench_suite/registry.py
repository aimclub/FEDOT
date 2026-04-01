from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .api import (
    run_forecasting_benchmark_suite,
    run_tsc_benchmark_suite,
    run_tser_benchmark_suite,
)
from .core import (
    BenchmarkSuiteConfig,
    ClassificationBenchmarkResult,
    ForecastingBenchmarkResult,
    RegressionBenchmarkResult,
    TaskType,
    ensure_directory,
    to_plain_data,
    write_json,
)
from .manifests import load_manifest, render_resolved_manifest, run_manifest
from .markdown import dataframe_to_markdown
from .presets import run_local_benchmark_preset

BenchmarkResult = any([ForecastingBenchmarkResult, ClassificationBenchmarkResult, RegressionBenchmarkResult])


@dataclass(frozen=True)
class BenchmarkRunBundle:
    result: BenchmarkResult
    run_dir: Path
    registry_entry_path: Path
    registry_index_path: Path
    summary_path: Path


def build_registry_entry(
        result: BenchmarkResult,
        *,
        execution_mode: str,
        source_path: str | None = None,
        input_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    successful_runs = sum(1 for record in result.run_records if record.status.value == 'success')
    failed_runs = sum(1 for record in result.run_records if record.status.value == 'failed')
    skipped_runs = sum(1 for record in result.run_records if record.status.value == 'skipped')
    not_available_runs = sum(1 for record in result.run_records if record.status.value == 'not_available')
    return {
        'run_id': result.run_id,
        'task_type': result.config.task_type.value,
        'execution_mode': execution_mode,
        'source_path': source_path or '',
        'output_dir': result.config.artifact_spec.output_dir,
        'primary_metric': result.aggregate_report.primary_metric,
        'dataset_count': len(result.config.datasets),
        'model_count': len(result.config.models),
        'artifact_count': len(result.artifact_manifest),
        'successful_runs': successful_runs,
        'failed_runs': failed_runs,
        'skipped_runs': skipped_runs,
        'not_available_runs': not_available_runs,
        'input_payload': to_plain_data(input_payload) if input_payload is not None else None,
    }


def persist_run_bundle(
        result: BenchmarkResult,
        *,
        execution_mode: str,
        source_path: str | None = None,
        input_payload: dict[str, Any] | None = None,
        resolved_payload: dict[str, Any] | None = None,
) -> BenchmarkRunBundle:
    base_dir = ensure_directory(result.config.artifact_spec.output_dir)
    run_dir = ensure_directory(base_dir / result.run_id)
    registry_dir = ensure_directory(base_dir / '_registry')

    entry = build_registry_entry(
        result,
        execution_mode=execution_mode,
        source_path=source_path,
        input_payload=input_payload,
    )
    entry['run_dir'] = str(run_dir)

    summary_path = run_dir / 'run_summary.json'
    write_json(summary_path, entry)

    resolved_config_path = run_dir / 'resolved_config.json'
    write_json(resolved_config_path, to_plain_data(result.config))

    if input_payload is not None:
        input_payload_path = run_dir / 'input_payload.json'
        write_json(input_payload_path, input_payload)

    if resolved_payload is not None:
        resolved_payload_path = run_dir / 'resolved_manifest.json'
        write_json(resolved_payload_path, resolved_payload)

    artifact_manifest_path = run_dir / 'artifact_manifest.json'
    write_json(artifact_manifest_path, [to_plain_data(record) for record in result.artifact_manifest])

    entry_path = registry_dir / f'{result.run_id}.json'
    write_json(entry_path, entry)

    index_path = registry_dir / 'run_registry.jsonl'
    with index_path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(to_plain_data(entry), ensure_ascii=False) + '\n')

    _write_registry_table(registry_dir)

    return BenchmarkRunBundle(
        result=result,
        run_dir=run_dir,
        registry_entry_path=entry_path,
        registry_index_path=index_path,
        summary_path=summary_path,
    )


def run_registered_manifest_path(path: str | Path) -> BenchmarkRunBundle:
    input_payload = load_manifest(path)
    resolved_payload = render_resolved_manifest(input_payload)
    result = run_manifest(input_payload)
    return persist_run_bundle(
        result,
        execution_mode='manifest',
        source_path=str(path),
        input_payload=input_payload,
        resolved_payload=resolved_payload,
    )


def run_registered_manifest(payload: dict[str, Any]) -> BenchmarkRunBundle:
    resolved_payload = render_resolved_manifest(payload)
    result = run_manifest(payload)
    return persist_run_bundle(
        result,
        execution_mode='manifest',
        input_payload=payload,
        resolved_payload=resolved_payload,
    )


def run_registered_suite(config: BenchmarkSuiteConfig) -> BenchmarkRunBundle:
    if config.task_type is TaskType.FORECASTING:
        result = run_forecasting_benchmark_suite(config)
    elif config.task_type is TaskType.TS_CLASSIFICATION:
        result = run_tsc_benchmark_suite(config)
    elif config.task_type is TaskType.TS_REGRESSION:
        result = run_tser_benchmark_suite(config)
    else:  # pragma: no cover
        raise ValueError(f'Unsupported task type: {config.task_type}')

    return persist_run_bundle(
        result,
        execution_mode='suite',
        resolved_payload=to_plain_data(config),
    )


def run_registered_preset(
        preset_name: str,
        *,
        dataset_name: str | None = None,
        subset: str | None = None,
        sample_size: int | None = None,
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        random_seed: int = 42,
        include_optional_external: bool = False,
        models=None,
) -> BenchmarkRunBundle:
    result = run_local_benchmark_preset(
        preset_name,
        dataset_name=dataset_name,
        subset=subset,
        sample_size=sample_size,
        output_dir=output_dir,
        persist_on_run=persist_on_run,
        random_seed=random_seed,
        include_optional_external=include_optional_external,
        models=models,
    )
    payload = {
        'version': 'benchmark_v2_manifest@1',
        'kind': 'preset',
        'preset_name': preset_name,
        'dataset_name': dataset_name,
        'subset': subset,
        'sample_size': sample_size,
        'output_dir': output_dir,
        'persist_on_run': persist_on_run,
        'random_seed': random_seed,
        'include_optional_external': include_optional_external,
        'models': to_plain_data(models) if models is not None else None,
    }
    return persist_run_bundle(
        result,
        execution_mode='preset',
        input_payload=payload,
        resolved_payload=payload,
    )


def _write_registry_table(registry_dir: Path) -> None:
    rows = []
    for entry_path in sorted(registry_dir.glob('*.json')):
        if entry_path.name == 'run_registry.json':
            continue
        rows.append(json.loads(entry_path.read_text(encoding='utf-8')))
    if not rows:
        return
    frame = pd.DataFrame(rows)
    csv_path = registry_dir / 'run_registry.csv'
    frame.to_csv(csv_path, index=False)
    markdown_path = registry_dir / 'run_registry.md'
    markdown_path.write_text(dataframe_to_markdown(frame, index=False), encoding='utf-8')
