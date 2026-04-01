import pandas as pd

from fedot.industrial.benchmark.tabular_tensor_suite import DEFAULT_MODES, render_markdown_report
from fedot.industrial.benchmark.tabular_tensor_suite.core import (
    ExecutorKind,
    GenerationBenchmarkRecord,
    LoadedTabularDataset,
    RunStatus,
    SuiteCapabilities,
    TabularDatasetSpec,
    TensorBenchmarkConfig,
    TensorBenchmarkSuiteResult,
)
from fedot.industrial.benchmark.tabular_tensor_suite.datasets import resolve_dataset_specs
from fedot.industrial.benchmark.tabular_tensor_suite.runner import run_tabular_tensor_suite


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
    assert 'Гипотеза 1' in report


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
    assert all('missing dependency golem' in record.skip_reason for record in result.generation_records)