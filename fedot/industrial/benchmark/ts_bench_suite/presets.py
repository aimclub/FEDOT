from __future__ import annotations

from pathlib import Path

from .api import run_forecasting_benchmark_suite, run_tsc_benchmark_suite, run_tser_benchmark_suite
from .core import ArtifactSpec, BenchmarkSuiteConfig, DatasetSpec, ModelSpec, RunSpec, TaskType

DEFAULT_PRESET_OUTPUT_DIR = Path('benchmark/results/v2_presets')


class BenchmarkPresetError(ValueError):
    pass


def build_local_m4_suite_config(
        *,
        subset: str = 'daily',
        sample_size: int | None = 3,
        random_seed: int = 42,
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        models: tuple[ModelSpec, ...] | None = None,
        include_optional_external: bool = False,
) -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='m4',
                dataset_name=f'm4_{subset.lower()}_local',
                subset=subset,
                sample_size=sample_size,
                random_seed=random_seed,
                adapter_options={'use_local_files': True},
            ),
        ),
        models=models or _default_forecasting_models(include_optional_external=include_optional_external),
        metrics=('mase', 'smape', 'owa', 'rmse', 'mae'),
        artifact_spec=_artifact_spec(output_dir, persist_on_run, 'm4'),
        run_spec=RunSpec(run_name=f'm4_{subset.lower()}_suite', primary_metric='mae'),
    )


def build_local_monash_suite_config(
        *,
        dataset_name: str = 'Bitcoin',
        subset: str = 'daily',
        sample_size: int | None = 3,
        random_seed: int = 42,
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        models: tuple[ModelSpec, ...] | None = None,
        include_optional_external: bool = False,
) -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='monash',
                dataset_name=dataset_name,
                subset=subset,
                sample_size=sample_size,
                random_seed=random_seed,
                adapter_options={'use_local_files': True},
            ),
        ),
        models=models or _default_forecasting_models(include_optional_external=include_optional_external),
        metrics=('mase', 'smape', 'owa', 'rmse', 'mae'),
        artifact_spec=_artifact_spec(output_dir, persist_on_run, 'monash'),
        run_spec=RunSpec(run_name=f'monash_{dataset_name.lower()}_suite', primary_metric='mae'),
    )


def build_local_ucr_suite_config(
        *,
        dataset_name: str = 'Lightning7',
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        models: tuple[ModelSpec, ...] | None = None,
) -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        task_type=TaskType.TS_CLASSIFICATION,
        datasets=(DatasetSpec(benchmark='ucr', dataset_name=dataset_name),),
        models=models or _default_classification_models(),
        metrics=('accuracy', 'balanced_accuracy', 'f1_macro'),
        artifact_spec=_artifact_spec(output_dir, persist_on_run, 'ucr'),
        run_spec=RunSpec(run_name=f'ucr_{dataset_name.lower()}_suite', primary_metric='accuracy'),
    )


def build_local_tser_suite_config(
        *,
        dataset_name: str = 'NaturalGasPricesSentiment',
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        models: tuple[ModelSpec, ...] | None = None,
) -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        task_type=TaskType.TS_REGRESSION,
        datasets=(DatasetSpec(benchmark='local_tser', dataset_name=dataset_name),),
        models=models or _default_regression_models(),
        metrics=('rmse', 'mae', 'r2'),
        artifact_spec=_artifact_spec(output_dir, persist_on_run, 'tser'),
        run_spec=RunSpec(run_name=f'tser_{dataset_name.lower()}_suite', primary_metric='rmse'),
    )


def run_local_benchmark_preset(
        preset_name: str,
        *,
        dataset_name: str | None = None,
        subset: str | None = None,
        sample_size: int | None = None,
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        random_seed: int = 42,
        include_optional_external: bool = False,
        models: tuple[ModelSpec, ...] | None = None,
):
    normalized = preset_name.lower()
    if normalized == 'm4':
        config = build_local_m4_suite_config(
            subset=subset or 'daily',
            sample_size=sample_size if sample_size is not None else 3,
            random_seed=random_seed,
            output_dir=output_dir,
            persist_on_run=persist_on_run,
            models=models,
            include_optional_external=include_optional_external,
        )
        return run_forecasting_benchmark_suite(config)
    if normalized == 'monash':
        config = build_local_monash_suite_config(
            dataset_name=dataset_name or 'Bitcoin',
            subset=subset or 'daily',
            sample_size=sample_size if sample_size is not None else 3,
            random_seed=random_seed,
            output_dir=output_dir,
            persist_on_run=persist_on_run,
            models=models,
            include_optional_external=include_optional_external,
        )
        return run_forecasting_benchmark_suite(config)
    if normalized == 'ucr':
        config = build_local_ucr_suite_config(
            dataset_name=dataset_name or 'Lightning7',
            output_dir=output_dir,
            persist_on_run=persist_on_run,
            models=models,
        )
        return run_tsc_benchmark_suite(config)
    if normalized == 'tser':
        config = build_local_tser_suite_config(
            dataset_name=dataset_name or 'NaturalGasPricesSentiment',
            output_dir=output_dir,
            persist_on_run=persist_on_run,
            models=models,
        )
        return run_tser_benchmark_suite(config)
    raise BenchmarkPresetError(f'Unsupported local benchmark preset: {preset_name}')


def _artifact_spec(
        output_dir: str | Path | None,
        persist_on_run: bool,
        preset_name: str,
) -> ArtifactSpec:
    return ArtifactSpec(
        output_dir=str(Path(output_dir) if output_dir is not None else DEFAULT_PRESET_OUTPUT_DIR / preset_name),
        persist_on_run=persist_on_run,
    )


def _default_forecasting_models(*, include_optional_external: bool) -> tuple[ModelSpec, ...]:
    models = [
        ModelSpec(adapter_name='naive_last_value', display_name='NaiveLastValue'),
        ModelSpec(adapter_name='moving_average', display_name='MovingAverage', params={'window_size': 3}),
        ModelSpec(adapter_name='linear_trend', display_name='LinearTrend'),
        ModelSpec(
            adapter_name='okhs',
            display_name='OKHS DMD',
            params={'method': 'dmd', 'window_size': 8, 'n_modes': 2, 'q': 0.9},
        ),
    ]
    if include_optional_external:
        models.extend(
            [
                ModelSpec(adapter_name='autogluon', display_name='AutoGluon', optional=True),
                ModelSpec(adapter_name='nbeats', display_name='N-BEATS', optional=True),
                ModelSpec(adapter_name='tft', display_name='TFT', optional=True),
            ]
        )
    return tuple(models)


def _default_classification_models() -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(adapter_name='majority_class', display_name='MajorityClass'),
        ModelSpec(adapter_name='nearest_centroid', display_name='NearestCentroid'),
    )


def _default_regression_models() -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(adapter_name='mean_regressor', display_name='MeanRegressor'),
        ModelSpec(adapter_name='linear_regressor', display_name='LinearRegressor'),
    )
