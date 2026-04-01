from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from fedot.core.optimisers.objective.runtime_data_rules import RuntimeEvaluationRecord

from .core import (
    ExecutorKind,
    GenerationBenchmarkRecord,
    IndividualBenchmarkRecord,
    RunStatus,
    SuiteCapabilities,
    TensorBenchmarkConfig,
    TensorBenchmarkSuiteResult,
    ensure_output_dir,
    new_run_id,
    parse_csv_tuple,
    safe_mean,
    safe_percentile,
    write_csv_records,
    write_json,
)
from .datasets import DATASET_REGISTRY, load_dataset, resolve_dataset_specs
from .report import render_markdown_report

DEFAULT_DATASETS = tuple(DATASET_REGISTRY.keys())
DEFAULT_MODES = ('input_cpu', 'tensor_cpu', 'input_gpu_bridge', 'tensor_gpu_bridge')


def build_config(datasets: tuple[str, ...] = DEFAULT_DATASETS,
                 modes: tuple[str, ...] = DEFAULT_MODES,
                 executor: str = 'sequential',
                 seeds: tuple[int, ...] = (42, 43, 44),
                 generations: int = 1,
                 pop_size: int = 15,
                 output_dir: str = 'benchmark_results/tabular_tensor_suite') -> TensorBenchmarkConfig:
    return TensorBenchmarkConfig(
        datasets=datasets,
        modes=modes,
        executor=ExecutorKind(executor),
        seeds=seeds,
        generations=generations,
        pop_size=pop_size,
        output_dir=output_dir,
    )


def detect_capabilities() -> SuiteCapabilities:
    gpu_available = False
    visible_gpu_count = 0
    gpu_reason = ''
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
        visible_gpu_count = int(torch.cuda.device_count()) if gpu_available else 0
        if not gpu_available:
            gpu_reason = 'CUDA unavailable in current environment.'
    except Exception as ex:
        gpu_reason = f'PyTorch/CUDA check failed: {ex}'

    dask_available = False
    dask_reason = ''
    try:
        from distributed import Client  # noqa: F401
        dask_available = True
    except Exception as ex:
        dask_reason = f'Dask distributed unavailable: {ex}'

    fedot_runtime_available = False
    fedot_runtime_reason = ''
    try:
        from fedot.api.main import Fedot  # noqa: F401

        fedot_runtime_available = True
    except ModuleNotFoundError as ex:
        missing_dependency = ex.name or 'unknown dependency'
        fedot_runtime_reason = f'FEDOT runtime unavailable: missing dependency {missing_dependency}.'
    except Exception as ex:
        fedot_runtime_reason = f'FEDOT runtime unavailable: {ex}'

    return SuiteCapabilities(
        gpu_available=gpu_available,
        dask_available=dask_available,
        visible_gpu_count=visible_gpu_count,
        fedot_runtime_available=fedot_runtime_available,
        gpu_reason=gpu_reason,
        dask_reason=dask_reason,
        fedot_runtime_reason=fedot_runtime_reason,
    )


def run_tabular_tensor_suite(config: TensorBenchmarkConfig) -> TensorBenchmarkSuiteResult:
    capabilities = detect_capabilities()
    output_dir = ensure_output_dir(config.output_dir)
    run_id = new_run_id('tabular_tensor_suite')

    individual_records: list[IndividualBenchmarkRecord] = []
    generation_records: list[GenerationBenchmarkRecord] = []

    for dataset_spec in resolve_dataset_specs(config.datasets):
        for seed in config.seeds:
            try:
                loaded_dataset = load_dataset(dataset_spec, seed=seed)
            except Exception as ex:
                generation_records.extend(
                    _build_dataset_load_skip_records(
                        dataset_name=dataset_spec.name,
                        problem=dataset_spec.problem,
                        seed=seed,
                        modes=config.modes,
                        executor=config.executor,
                        reason=f'Dataset load failed: {ex}',
                    )
                )
                continue

            runtime_skip_reason = _resolve_runtime_skip_reason(capabilities)
            if runtime_skip_reason is not None:
                generation_records.extend(
                    _build_dataset_load_skip_records(
                        dataset_name=dataset_spec.name,
                        problem=dataset_spec.problem,
                        seed=seed,
                        modes=config.modes,
                        executor=config.executor,
                        reason=runtime_skip_reason,
                        size_bucket=loaded_dataset.size_bucket,
                    )
                )
                continue

            split_payload = _build_dataset_split(loaded_dataset, seed=seed)
            job_specs = [
                dict(
                    split_payload=split_payload,
                    mode=mode,
                    seed=seed,
                    config=config,
                    capabilities=capabilities,
                )
                for mode in config.modes
            ]
            job_results = _execute_jobs(job_specs, executor=config.executor, capabilities=capabilities)
            for generation_record, run_individual_records in job_results:
                generation_records.append(generation_record)
                individual_records.extend(run_individual_records)

    report_path = output_dir / 'report.md'
    config_path = output_dir / 'resolved_config.json'
    capabilities_path = output_dir / 'capabilities.json'
    individual_json_path = output_dir / 'per_individual.json'
    generation_json_path = output_dir / 'per_generation.json'
    individual_csv_path = output_dir / 'per_individual.csv'
    generation_csv_path = output_dir / 'per_generation.csv'

    result = TensorBenchmarkSuiteResult(
        run_id=run_id,
        config=config,
        capabilities=capabilities,
        individual_records=tuple(individual_records),
        generation_records=tuple(generation_records),
        artifact_paths=(
            str(config_path),
            str(capabilities_path),
            str(individual_json_path),
            str(generation_json_path),
            str(individual_csv_path),
            str(generation_csv_path),
            str(report_path),
        ),
    )

    write_json(config_path, config)
    write_json(capabilities_path, capabilities)
    write_json(individual_json_path, result.individual_records)
    write_json(generation_json_path, result.generation_records)
    write_csv_records(individual_csv_path, result.individual_records)
    write_csv_records(generation_csv_path, result.generation_records)
    report_path.write_text(render_markdown_report(result), encoding='utf-8')
    return result


def _execute_jobs(job_specs: list[dict], executor: ExecutorKind, capabilities: SuiteCapabilities):
    if executor is ExecutorKind.SEQUENTIAL:
        return [_run_mode_job(**job_spec) for job_spec in job_specs]

    if not capabilities.dask_available:
        return [_run_mode_job(**job_spec) for job_spec in job_specs]

    from distributed import Client, LocalCluster

    worker_count = 1 if any(job_spec['mode'].endswith('gpu_bridge') for job_spec in job_specs) else min(4, len(job_specs))
    cluster = LocalCluster(processes=False, n_workers=worker_count, threads_per_worker=1)
    client = Client(cluster)
    try:
        futures = [client.submit(_run_mode_job, **job_spec) for job_spec in job_specs]
        return client.gather(futures)
    finally:
        client.close()
        cluster.close()


def _run_mode_job(split_payload: dict,
                  mode: str,
                  seed: int,
                  config: TensorBenchmarkConfig,
                  capabilities: SuiteCapabilities):
    loaded_dataset = split_payload['loaded_dataset']
    problem = loaded_dataset.spec.problem
    size_bucket = loaded_dataset.size_bucket
    executor_name = config.executor.value
    device = 'gpu' if mode.endswith('gpu_bridge') else 'cpu'
    data_mode = 'tensor' if mode.startswith('tensor_') else 'input'

    skip_reason = _resolve_skip_reason(mode=mode, executor=config.executor, capabilities=capabilities)
    if skip_reason is not None:
        return (
            GenerationBenchmarkRecord(
                dataset_name=loaded_dataset.spec.name,
                problem=problem,
                size_bucket=size_bucket,
                mode=mode,
                seed=seed,
                executor=executor_name,
                status=RunStatus.SKIPPED.value,
                skip_reason=skip_reason,
                device=device,
                data_mode=data_mode,
                quality_metric_name=_quality_metric_name(problem),
                quality_metric_value=None,
                wall_clock_sec=None,
                mean_fit_sec=None,
                median_fit_sec=None,
                p95_fit_sec=None,
                success_rate=None,
                throughput_individuals_per_min=None,
                best_metric=None,
                top3_mean_metric=None,
                unique_pipeline_ratio=None,
            ),
            [],
        )

    run_started_at = perf_counter()
    _reset_gpu_peak_if_available()
    fedot_model = _build_fedot_model(problem=problem, mode=mode, seed=seed, config=config)

    try:
        if data_mode == 'input':
            fedot_model.fit(features=split_payload['train_input'], predefined_model=None)
        else:
            train_tensor = fedot_model.data_processor.to_tensordata(
                split_payload['train_input'],
                backend_name=_tensor_backend_name(mode),
                is_predict=False,
            )
            fedot_model.fit_tensordata(tensor_data=train_tensor, predefined_model='auto')

        quality_value = _evaluate_holdout_quality(
            fedot_model=fedot_model,
            split_payload=split_payload,
            mode=mode,
            problem=problem,
        )
        status = RunStatus.SUCCESS.value
        run_individual_records = _build_individual_records(
            dataset_name=loaded_dataset.spec.name,
            problem=problem,
            size_bucket=size_bucket,
            mode=mode,
            seed=seed,
            executor=executor_name,
            device=device,
            data_mode=data_mode,
            evaluation_records=fedot_model.api_composer.last_evaluation_records,
        )
    except Exception as ex:
        quality_value = None
        status = RunStatus.FAILED.value
        run_individual_records = [
            IndividualBenchmarkRecord(
                dataset_name=loaded_dataset.spec.name,
                problem=problem,
                size_bucket=size_bucket,
                mode=mode,
                seed=seed,
                executor=executor_name,
                pipeline_id='run_failed',
                n_nodes=0,
                n_model_nodes=0,
                fit_time_sec=0.0,
                predict_time_sec=0.0,
                objective_value=None,
                success=False,
                failure_reason=str(ex),
                device=device,
                data_mode=data_mode,
                cpu_mem_mb=_get_cpu_mem_mb(),
                gpu_mem_mb_peak=_get_gpu_peak_mb_if_available(),
            )
        ]

    wall_clock_sec = perf_counter() - run_started_at
    generation_record = _aggregate_generation_record(
        dataset_name=loaded_dataset.spec.name,
        problem=problem,
        size_bucket=size_bucket,
        mode=mode,
        seed=seed,
        executor=executor_name,
        status=status,
        skip_reason='',
        device=device,
        data_mode=data_mode,
        quality_metric_name=_quality_metric_name(problem),
        quality_metric_value=quality_value,
        wall_clock_sec=wall_clock_sec,
        individual_records=run_individual_records,
    )
    return generation_record, run_individual_records


def _build_dataset_split(loaded_dataset, seed: int):
    stratify_target = loaded_dataset.target if loaded_dataset.spec.problem == 'classification' else None
    train_features, test_features, train_target, test_target = train_test_split(
        loaded_dataset.features,
        loaded_dataset.target,
        test_size=loaded_dataset.spec.test_size,
        random_state=seed,
        stratify=stratify_target,
    )
    prepared_train_target = train_target.reset_index(drop=True) if isinstance(train_target, pd.Series) else train_target
    prepared_test_target = test_target.reset_index(drop=True) if isinstance(test_target, pd.Series) else test_target
    train_features = train_features.reset_index(drop=True)
    test_features = test_features.reset_index(drop=True)

    from fedot.api.main import Fedot

    bootstrap_model = Fedot(problem=loaded_dataset.spec.problem, timeout=1.0, with_tuning=False, show_progress=False)
    train_input = bootstrap_model.data_processor.define_data(features=train_features, target=prepared_train_target, is_predict=False)
    test_input = bootstrap_model.data_processor.define_data(features=test_features, target=prepared_test_target, is_predict=True)
    return {
        'loaded_dataset': loaded_dataset,
        'train_input': train_input,
        'test_input': test_input,
        'train_target': np.array(prepared_train_target),
        'test_target': np.array(prepared_test_target),
    }


def _build_fedot_model(problem: str, mode: str, seed: int, config: TensorBenchmarkConfig):
    from fedot.api.main import Fedot

    return Fedot(
        problem=problem,
        timeout=config.timeout_minutes,
        seed=seed,
        n_jobs=1,
        with_tuning=False,
        show_progress=False,
        cv_folds=config.cv_folds,
        pop_size=config.pop_size,
        num_of_generations=config.generations,
        available_operations=list(_resolve_operation_bundle(problem, mode)),
        metric=_quality_metric_name(problem),
        preset='best_quality',
        benchmark_runtime_mode=mode,
        benchmark_tensor_backend_name=_tensor_backend_name(mode),
    )


def _resolve_operation_bundle(problem: str, mode: str) -> tuple[str, ...]:
    if mode.endswith('gpu_bridge'):
        if problem == 'classification':
            return ('logit', 'rf', 'industrial_inception_nn', 'industrial_resnet_nn')
        return ('ridge', 'dtreg', 'industrial_inception_nn', 'industrial_resnet_nn')

    if problem == 'classification':
        return ('logit', 'rf', 'dt', 'scaling')
    return ('ridge', 'lasso', 'dtreg', 'scaling')


def _resolve_skip_reason(mode: str, executor: ExecutorKind, capabilities: SuiteCapabilities) -> str | None:
    if mode.endswith('gpu_bridge') and not capabilities.gpu_available:
        return capabilities.gpu_reason or 'GPU режим недоступен.'
    if executor is ExecutorKind.DASK_EXPERIMENTAL and not capabilities.dask_available:
        return capabilities.dask_reason or 'Dask experimental executor недоступен.'
    return None


def _evaluate_holdout_quality(fedot_model: Fedot, split_payload: dict, mode: str, problem: str) -> float:
    if problem == 'classification':
        if mode.startswith('tensor_'):
            test_tensor = fedot_model.data_processor.to_tensordata(
                split_payload['test_input'],
                backend_name=_tensor_backend_name(mode),
                is_predict=True,
            )
            probabilities = fedot_model.predict_proba_tensordata(test_tensor, probs_for_all_classes=True)
        else:
            probabilities = fedot_model.predict_proba(split_payload['test_input'], probs_for_all_classes=True)

        y_true = split_payload['test_target']
        unique_count = len(np.unique(y_true))
        if unique_count > 2:
            return float(roc_auc_score(y_true, probabilities, multi_class='ovr', average='macro'))
        probability_vector = probabilities[:, 1] if np.ndim(probabilities) > 1 else probabilities
        return float(roc_auc_score(y_true, probability_vector))

    if mode.startswith('tensor_'):
        test_tensor = fedot_model.data_processor.to_tensordata(
            split_payload['test_input'],
            backend_name=_tensor_backend_name(mode),
            is_predict=True,
        )
        prediction = fedot_model.predict_tensordata(test_tensor)
    else:
        prediction = fedot_model.predict(split_payload['test_input'])

    return float(mean_squared_error(split_payload['test_target'], prediction, squared=False))


def _build_individual_records(dataset_name: str,
                              problem: str,
                              size_bucket: str,
                              mode: str,
                              seed: int,
                              executor: str,
                              device: str,
                              data_mode: str,
                              evaluation_records: Iterable[RuntimeEvaluationRecord]):
    records = []
    for evaluation_record in evaluation_records:
        objective_value = evaluation_record.objective_values[0] if evaluation_record.objective_values else None
        records.append(
            IndividualBenchmarkRecord(
                dataset_name=dataset_name,
                problem=problem,
                size_bucket=size_bucket,
                mode=mode,
                seed=seed,
                executor=executor,
                pipeline_id=evaluation_record.pipeline_id,
                n_nodes=evaluation_record.n_nodes,
                n_model_nodes=evaluation_record.n_model_nodes,
                fit_time_sec=evaluation_record.fit_time_sec,
                predict_time_sec=evaluation_record.predict_time_sec,
                objective_value=_transform_objective_value(problem, objective_value),
                success=evaluation_record.success,
                failure_reason='' if evaluation_record.success else 'objective evaluation failed',
                device=device,
                data_mode=data_mode,
                cpu_mem_mb=_get_cpu_mem_mb(),
                gpu_mem_mb_peak=_get_gpu_peak_mb_if_available(),
            )
        )
    return records


def _aggregate_generation_record(dataset_name: str,
                                 problem: str,
                                 size_bucket: str,
                                 mode: str,
                                 seed: int,
                                 executor: str,
                                 status: str,
                                 skip_reason: str,
                                 device: str,
                                 data_mode: str,
                                 quality_metric_name: str,
                                 quality_metric_value: float | None,
                                 wall_clock_sec: float,
                                 individual_records: list[IndividualBenchmarkRecord]) -> GenerationBenchmarkRecord:
    successful_records = [record for record in individual_records if record.success]
    fit_times = [record.fit_time_sec for record in successful_records]
    transformed_objectives = [record.objective_value for record in successful_records if record.objective_value is not None]

    if problem == 'classification':
        best_objective = max(transformed_objectives) if transformed_objectives else None
        top3_values = sorted(transformed_objectives, reverse=True)[:3]
    else:
        best_objective = min(transformed_objectives) if transformed_objectives else None
        top3_values = sorted(transformed_objectives)[:3]

    success_rate = (len(successful_records) / len(individual_records)) if individual_records else None
    throughput = (len(successful_records) / wall_clock_sec * 60.0) if wall_clock_sec and successful_records else None
    unique_pipeline_ratio = (
        len({record.pipeline_id for record in successful_records}) / len(successful_records)
        if successful_records else None
    )

    return GenerationBenchmarkRecord(
        dataset_name=dataset_name,
        problem=problem,
        size_bucket=size_bucket,
        mode=mode,
        seed=seed,
        executor=executor,
        status=status,
        skip_reason=skip_reason,
        device=device,
        data_mode=data_mode,
        quality_metric_name=quality_metric_name,
        quality_metric_value=quality_metric_value,
        wall_clock_sec=wall_clock_sec,
        mean_fit_sec=safe_mean(fit_times),
        median_fit_sec=safe_percentile(fit_times, 50),
        p95_fit_sec=safe_percentile(fit_times, 95),
        success_rate=success_rate,
        throughput_individuals_per_min=throughput,
        best_metric=best_objective if best_objective is not None else quality_metric_value,
        top3_mean_metric=safe_mean(top3_values),
        unique_pipeline_ratio=unique_pipeline_ratio,
    )


def _build_dataset_load_skip_records(dataset_name: str,
                                     problem: str,
                                     seed: int,
                                     modes: tuple[str, ...],
                                     executor: ExecutorKind,
                                     reason: str,
                                     size_bucket: str = 'unknown'):
    records = []
    for mode in modes:
        records.append(
            GenerationBenchmarkRecord(
                dataset_name=dataset_name,
                problem=problem,
                size_bucket=size_bucket,
                mode=mode,
                seed=seed,
                executor=executor.value,
                status=RunStatus.SKIPPED.value,
                skip_reason=reason,
                device='gpu' if mode.endswith('gpu_bridge') else 'cpu',
                data_mode='tensor' if mode.startswith('tensor_') else 'input',
                quality_metric_name=_quality_metric_name(problem),
                quality_metric_value=None,
                wall_clock_sec=None,
                mean_fit_sec=None,
                median_fit_sec=None,
                p95_fit_sec=None,
                success_rate=None,
                throughput_individuals_per_min=None,
                best_metric=None,
                top3_mean_metric=None,
                unique_pipeline_ratio=None,
            )
        )
    return records


def _quality_metric_name(problem: str) -> str:
    return 'roc_auc' if problem == 'classification' else 'rmse'


def _resolve_runtime_skip_reason(capabilities: SuiteCapabilities) -> str | None:
    if capabilities.fedot_runtime_available:
        return None
    return capabilities.fedot_runtime_reason or 'FEDOT runtime unavailable.'


def _tensor_backend_name(mode: str) -> str:
    return 'gpu' if mode.endswith('gpu_bridge') else 'cpu'


def _transform_objective_value(problem: str, objective_value: float | None) -> float | None:
    if objective_value is None:
        return None
    if problem == 'classification':
        return float(-objective_value)
    return float(objective_value)


def _get_cpu_mem_mb() -> float | None:
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / (1024 ** 2))
    except Exception:
        return None


def _reset_gpu_peak_if_available() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return None


def _get_gpu_peak_mb_if_available() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    except Exception:
        return None
    return None


__all__ = [
    'DEFAULT_DATASETS',
    'DEFAULT_MODES',
    'build_config',
    'detect_capabilities',
    'run_tabular_tensor_suite',
]




