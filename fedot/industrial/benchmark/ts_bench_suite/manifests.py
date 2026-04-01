from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .api import run_forecasting_benchmark_suite, run_tsc_benchmark_suite, run_tser_benchmark_suite
from .core import ArtifactSpec, BenchmarkSuiteConfig, DatasetSpec, ModelSpec, RunSpec, TaskType, to_plain_data
from .presets import run_local_benchmark_preset

MANIFEST_VERSION = 'benchmark_v2_manifest@1'


class BenchmarkManifestError(ValueError):
    pass


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise BenchmarkManifestError(f'Manifest file does not exist: {manifest_path}')

    suffix = manifest_path.suffix.lower()
    if suffix == '.json':
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    elif suffix in {'.yml', '.yaml'}:
        payload = yaml.safe_load(manifest_path.read_text(encoding='utf-8'))
    else:
        raise BenchmarkManifestError(f'Unsupported manifest format: {suffix}')

    if not isinstance(payload, dict):
        raise BenchmarkManifestError('Manifest root must be a mapping/object.')
    return payload


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    version = str(payload.get('version', MANIFEST_VERSION))
    if version != MANIFEST_VERSION:
        raise BenchmarkManifestError(f'Unsupported manifest version: {version}')

    kind = str(payload.get('kind', '')).lower()
    if kind not in {'preset', 'suite'}:
        raise BenchmarkManifestError('Manifest kind must be "preset" or "suite".')

    if kind == 'preset':
        if 'preset_name' not in payload:
            raise BenchmarkManifestError('Preset manifest must define "preset_name".')
        return payload

    if 'task_type' not in payload:
        raise BenchmarkManifestError('Suite manifest must define "task_type".')
    if 'datasets' not in payload or not payload['datasets']:
        raise BenchmarkManifestError('Suite manifest must define a non-empty "datasets" list.')
    if 'models' not in payload or not payload['models']:
        raise BenchmarkManifestError('Suite manifest must define a non-empty "models" list.')
    return payload


def build_suite_config_from_manifest(payload: dict[str, Any]) -> BenchmarkSuiteConfig:
    validated = validate_manifest(payload)
    if str(validated['kind']).lower() != 'suite':
        raise BenchmarkManifestError('build_suite_config_from_manifest expects kind="suite".')

    artifact_payload = dict(validated.get('artifact_spec', {}))
    run_payload = dict(validated.get('run_spec', {}))
    metrics = tuple(validated.get('metrics', _default_metrics_for_task(validated['task_type'])))

    if 'output_dir' not in artifact_payload:
        raise BenchmarkManifestError('Suite manifest artifact_spec must define "output_dir".')

    return BenchmarkSuiteConfig(
        task_type=TaskType(validated['task_type']),
        datasets=tuple(DatasetSpec(**dataset_payload) for dataset_payload in validated['datasets']),
        models=tuple(ModelSpec(**model_payload) for model_payload in validated['models']),
        artifact_spec=ArtifactSpec(**artifact_payload),
        run_spec=RunSpec(**run_payload) if run_payload else RunSpec(primary_metric=metrics[0]),
        metrics=metrics,
    )


def run_manifest(payload: dict[str, Any]):
    validated = validate_manifest(payload)
    kind = str(validated['kind']).lower()
    if kind == 'preset':
        model_specs = tuple(ModelSpec(**item) for item in validated.get('models', ())) or None
        return run_local_benchmark_preset(
            validated['preset_name'],
            dataset_name=validated.get('dataset_name'),
            subset=validated.get('subset'),
            sample_size=validated.get('sample_size'),
            output_dir=validated.get('output_dir'),
            persist_on_run=bool(validated.get('persist_on_run', True)),
            random_seed=int(validated.get('random_seed', 42)),
            include_optional_external=bool(validated.get('include_optional_external', False)),
            models=model_specs,
        )

    config = build_suite_config_from_manifest(validated)
    if config.task_type is TaskType.FORECASTING:
        return run_forecasting_benchmark_suite(config)
    if config.task_type is TaskType.TS_CLASSIFICATION:
        return run_tsc_benchmark_suite(config)
    if config.task_type is TaskType.TS_REGRESSION:
        return run_tser_benchmark_suite(config)
    raise BenchmarkManifestError(f'Unsupported task type in manifest: {config.task_type}')


def run_manifest_path(path: str | Path):
    return run_manifest(load_manifest(path))


def render_resolved_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    validated = validate_manifest(payload)
    if str(validated['kind']).lower() == 'preset':
        return to_plain_data(validated)
    config = build_suite_config_from_manifest(validated)
    return to_plain_data(config)


def write_example_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = manifest_path.suffix.lower()
    if suffix == '.json':
        manifest_path.write_text(json.dumps(to_plain_data(payload), indent=2, ensure_ascii=False), encoding='utf-8')
        return
    if suffix in {'.yml', '.yaml'}:
        manifest_path.write_text(yaml.safe_dump(to_plain_data(payload), sort_keys=False, allow_unicode=True),
                                 encoding='utf-8')
        return
    raise BenchmarkManifestError(f'Unsupported manifest format for writing: {suffix}')


def _default_metrics_for_task(task_type: str | TaskType) -> tuple[str, ...]:
    task = TaskType(task_type)
    if task is TaskType.FORECASTING:
        return ('mase', 'smape', 'owa', 'rmse', 'mae')
    if task is TaskType.TS_CLASSIFICATION:
        return ('accuracy', 'balanced_accuracy', 'f1_macro')
    return ('rmse', 'mae', 'r2')
