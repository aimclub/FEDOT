from pathlib import Path

import pandas as pd

from fedot.industrial.benchmark.tabular_tensor_suite import DEFAULT_MODES, render_markdown_report
from fedot.industrial.benchmark.tabular_tensor_suite.core import (
    ExecutorKind,
    GenerationBenchmarkRecord,
    IndividualBenchmarkRecord,
    LoadedTabularDataset,
    RunStatus,
    SuiteCapabilities,
    TabularDatasetSpec,
    TensorBenchmarkConfig,
    TensorBenchmarkSuiteResult,
)
from fedot.industrial.benchmark.tabular_tensor_suite.datasets import resolve_dataset_specs
from fedot.industrial.benchmark.tabular_tensor_suite.runner import (
    _resolve_dask_worker_count,
    run_tabular_tensor_suite,
)



def test_tabular_tensor_suite_dataset_registry_and_defaults():
    config = TensorBenchmarkConfig(
        datasets=('kc2',),
        modes=('input_cpu',),
        executor=ExecutorKind.SEQUENTIAL,
        seeds=(42,),
    )
    specs = resolve_dataset_specs(config.datasets)

    assert DEFAULT_MODES[0] == 'input_cpu'
    assert specs[0].name == 'kc2'
    assert specs[0].problem == 'classification'
    assert config.show_progress is True



def test_tabular_tensor_suite_report_mentions_missing_dask():
    config = TensorBenchmarkConfig(
        datasets=('kc2',),
        modes=('input_cpu',),
        executor=ExecutorKind.SEQUENTIAL,
        seeds=(42,),
    )
    result = TensorBenchmarkSuiteResult(
        run_id='suite_test',
        config=config,
        capabilities=SuiteCapabilities(
            gpu_available=False,
            dask_available=False,
            visible_gpu_count=0,
            gpu_reason='CUDA unavailable',
            dask_reason='distributed missing',
        ),
        individual_records=tuple(),
        generation_records=(
            GenerationBenchmarkRecord(
                dataset_name='kc2',
                problem='classification',
                size_bucket='small',
                mode='input_cpu',
                seed=42,
                executor='sequential',
                status=RunStatus.SKIPPED.value,
                skip_reason='demo',
                failure_stage='validate',
                device='cpu',
                data_mode='input',
                quality_metric_name='roc_auc',
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
        ),
    )

    report = render_markdown_report(result)

    assert 'Dask' in report
    assert 'distributed missing' in report
    assert 'Р“РёРїРѕС‚РµР·Р° 1' in report
    assert 'stage=validate' in report



def test_tabular_tensor_suite_report_mentions_runtime_unavailable():
    config = TensorBenchmarkConfig(
        datasets=('kc2',),
        modes=('input_cpu',),
        executor=ExecutorKind.SEQUENTIAL,
        seeds=(42,),
    )
    result = TensorBenchmarkSuiteResult(
        run_id='suite_runtime_blocked',
        config=config,
        capabilities=SuiteCapabilities(
            gpu_available=False,
            dask_available=False,
            visible_gpu_count=0,
            fedot_runtime_available=False,
            fedot_runtime_reason='FEDOT runtime unavailable: missing dependency golem.',
        ),
        individual_records=tuple(),
        generation_records=tuple(),
    )

    report = render_markdown_report(result)

    assert 'FEDOT runtime available: `False`' in report
    assert 'missing dependency golem' in report



def test_run_tabular_tensor_suite_marks_runtime_missing_as_skipped(monkeypatch, tmp_path):
    config = TensorBenchmarkConfig(
        datasets=('kc2',),
        modes=('input_cpu', 'tensor_cpu'),
        executor=ExecutorKind.SEQUENTIAL,
        seeds=(42,),
        output_dir=str(tmp_path / 'tensor_suite'),
        show_progress=False,
    )
    loaded_dataset = LoadedTabularDataset(
        spec=TabularDatasetSpec(
            name='kc2',
            problem='classification',
            loader_kind='local_csv',
            target_column='target',
        ),
        features=pd.DataFrame({'f1': [0.0, 1.0, 2.0, 3.0]}),
        target=pd.Series([0, 1, 0, 1]),
        sample_count=4,
        feature_count=1,
        size_bucket='small',
        metadata={},
    )

    monkeypatch.setattr(
        'fedot.industrial.benchmark.tabular_tensor_suite.runner.detect_capabilities',
        lambda: SuiteCapabilities(
            gpu_available=False,
            dask_available=False,
            visible_gpu_count=0,
            fedot_runtime_available=False,
            fedot_runtime_reason='FEDOT runtime unavailable: missing dependency golem.',
        ),
    )
    monkeypatch.setattr(
        'fedot.industrial.benchmark.tabular_tensor_suite.runner.load_dataset',
        lambda spec, seed=42: loaded_dataset,
    )

    result = run_tabular_tensor_suite(config)

    assert len(result.individual_records) == 0
    assert len(result.generation_records) == 2
    assert all(record.status == RunStatus.SKIPPED.value for record in result.generation_records)
    assert all(record.size_bucket == 'small' for record in result.generation_records)
    assert all(record.failure_stage == 'validate' for record in result.generation_records)
    assert all('missing dependency golem' in record.skip_reason for record in result.generation_records)



def test_run_tabular_tensor_suite_continue_and_mark_preserves_failure_stage(monkeypatch, tmp_path):
    config = TensorBenchmarkConfig(
        datasets=('kc2',),
        modes=('input_cpu', 'tensor_cpu'),
        executor=ExecutorKind.SEQUENTIAL,
        seeds=(42,),
        output_dir=str(tmp_path / 'tensor_suite_continue'),
        show_progress=False,
    )
    loaded_dataset = LoadedTabularDataset(
        spec=TabularDatasetSpec(
            name='kc2',
            problem='classification',
            loader_kind='local_csv',
            target_column='target',
        ),
        features=pd.DataFrame({'f1': [0.0, 1.0, 2.0, 3.0]}),
        target=pd.Series([0, 1, 0, 1]),
        sample_count=4,
        feature_count=1,
        size_bucket='small',
        metadata={},
    )

    monkeypatch.setattr(
        'fedot.industrial.benchmark.tabular_tensor_suite.runner.detect_capabilities',
        lambda: SuiteCapabilities(
            gpu_available=False,
            dask_available=False,
            visible_gpu_count=0,
            fedot_runtime_available=True,
        ),
    )
    monkeypatch.setattr(
        'fedot.industrial.benchmark.tabular_tensor_suite.runner.load_dataset',
        lambda spec, seed=42: loaded_dataset,
    )
    monkeypatch.setattr(
        'fedot.industrial.benchmark.tabular_tensor_suite.runner._build_dataset_split',
        lambda dataset, seed=42: {'loaded_dataset': loaded_dataset},
    )

    def fake_run_mode_job(split_payload, mode, seed, config, capabilities, progress=None):
        if mode == 'input_cpu':
            return (
                GenerationBenchmarkRecord(
                    dataset_name='kc2',
                    problem='classification',
                    size_bucket='small',
                    mode=mode,
                    seed=seed,
                    executor=config.executor.value,
                    status=RunStatus.FAILED.value,
                    skip_reason='fit exploded',
                    failure_stage='fit',
                    device='cpu',
                    data_mode='input',
                    quality_metric_name='roc_auc',
                    quality_metric_value=None,
                    wall_clock_sec=1.0,
                    mean_fit_sec=None,
                    median_fit_sec=None,
                    p95_fit_sec=None,
                    success_rate=0.0,
                    throughput_individuals_per_min=None,
                    best_metric=None,
                    top3_mean_metric=None,
                    unique_pipeline_ratio=None,
                ),
                [
                    IndividualBenchmarkRecord(
                        dataset_name='kc2',
                        problem='classification',
                        size_bucket='small',
                        mode=mode,
                        seed=seed,
                        executor=config.executor.value,
                        pipeline_id='run_failed',
                        n_nodes=0,
                        n_model_nodes=0,
                        fit_time_sec=0.0,
                        predict_time_sec=0.0,
                        objective_value=None,
                        success=False,
                        failure_reason='fit exploded',
                        failure_stage='fit',
                        device='cpu',
                        data_mode='input',
                    )
                ],
            )
        return (
            GenerationBenchmarkRecord(
                dataset_name='kc2',
                problem='classification',
                size_bucket='small',
                mode=mode,
                seed=seed,
                executor=config.executor.value,
                status=RunStatus.SUCCESS.value,
                skip_reason='',
                failure_stage=None,
                device='cpu',
                data_mode='tensor',
                quality_metric_name='roc_auc',
                quality_metric_value=0.71,
                wall_clock_sec=1.5,
                mean_fit_sec=0.7,
                median_fit_sec=0.7,
                p95_fit_sec=0.7,
                success_rate=1.0,
                throughput_individuals_per_min=60.0,
                best_metric=0.71,
                top3_mean_metric=0.71,
                unique_pipeline_ratio=1.0,
            ),
            [
                IndividualBenchmarkRecord(
                    dataset_name='kc2',
                    problem='classification',
                    size_bucket='small',
                    mode=mode,
                    seed=seed,
                    executor=config.executor.value,
                    pipeline_id='pipeline_ok',
                    n_nodes=1,
                    n_model_nodes=1,
                    fit_time_sec=0.7,
                    predict_time_sec=0.1,
                    objective_value=0.71,
                    success=True,
                    failure_reason='',
                    failure_stage=None,
                    device='cpu',
                    data_mode='tensor',
                )
            ],
        )

    monkeypatch.setattr(
        'fedot.industrial.benchmark.tabular_tensor_suite.runner._run_mode_job',
        fake_run_mode_job,
    )

    result = run_tabular_tensor_suite(config)
    report_path = Path(result.artifact_paths[-1])
    report_text = report_path.read_text(encoding='utf-8')

    assert len(result.generation_records) == 2
    assert [record.status for record in result.generation_records] == [RunStatus.FAILED.value, RunStatus.SUCCESS.value]
    assert result.generation_records[0].failure_stage == 'fit'
    assert len(result.individual_records) == 2
    assert result.individual_records[0].failure_stage == 'fit'
    assert report_path.exists()
    assert 'fit exploded' in report_text



def test_resolve_dask_worker_count_limits_gpu_workers_to_visible_devices():
    capabilities = SuiteCapabilities(
        gpu_available=True,
        dask_available=True,
        visible_gpu_count=1,
    )

    worker_count = _resolve_dask_worker_count(
        job_specs=[
            {'mode': 'input_gpu_bridge'},
            {'mode': 'tensor_gpu_bridge'},
            {'mode': 'input_cpu'},
        ],
        capabilities=capabilities,
    )

    assert worker_count == 1



def test_gpu_bridge_modes_share_gpu_only_operation_bundle():
    from fedot.industrial.benchmark.tabular_tensor_suite.runner import _resolve_operation_bundle

    expected_bundle = ('industrial_inception_nn', 'industrial_resnet_nn')

    assert _resolve_operation_bundle('classification', 'input_gpu_bridge') == expected_bundle
    assert _resolve_operation_bundle('classification', 'tensor_gpu_bridge') == expected_bundle
    assert _resolve_operation_bundle('regression', 'input_gpu_bridge') == expected_bundle
    assert _resolve_operation_bundle('regression', 'tensor_gpu_bridge') == expected_bundle