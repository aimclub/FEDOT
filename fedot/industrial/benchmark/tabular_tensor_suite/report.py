from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .core import GenerationBenchmarkRecord, SuiteCapabilities, TensorBenchmarkConfig, TensorBenchmarkSuiteResult


CPU_MODE_PAIR = ('input_cpu', 'tensor_cpu')
GPU_MODE_PAIR = ('input_gpu_bridge', 'tensor_gpu_bridge')


def render_markdown_report(result: TensorBenchmarkSuiteResult) -> str:
    lines = [
        '# TensorData Benchmark Report',
        '',
        '## Сводка',
        '',
        f'- Run ID: `{result.run_id}`',
        f'- Datasets: {", ".join(result.config.datasets)}',
        f'- Modes: {", ".join(result.config.modes)}',
        f'- Executor: `{result.config.executor.value}`',
        f'- Seeds: {", ".join(str(seed) for seed in result.config.seeds)}',
        f'- FEDOT runtime available: `{result.capabilities.fedot_runtime_available}`',
        f'- GPU available: `{result.capabilities.gpu_available}`',
        f'- Dask available: `{result.capabilities.dask_available}`',
        '',
    ]

    if not result.capabilities.fedot_runtime_available:
        lines.extend([
            f'- FEDOT runtime unavailable: {_ensure_terminal_period(result.capabilities.fedot_runtime_reason or "не удалось импортировать runtime зависимости")}',
            '',
        ])

    lines.extend([
        '## Гипотеза 1. Ускорение fit одного индивида',
        '',
    ])
    lines.extend(_render_fit_hypothesis(result.generation_records, CPU_MODE_PAIR, title='CPU path'))
    lines.extend(_render_fit_hypothesis(result.generation_records, GPU_MODE_PAIR, title='GPU bridge path'))

    lines.extend([
        '',
        '## Гипотеза 2. Ускорение generation wall time и search efficiency',
        '',
    ])
    lines.extend(_render_generation_hypothesis(result.generation_records))

    lines.extend([
        '',
        '## Гипотеза 3. Эффект experimental Dask backend',
        '',
    ])
    lines.extend(_render_dask_hypothesis(result.generation_records, result.capabilities))

    return '\n'.join(lines).strip() + '\n'


def _render_fit_hypothesis(records: Iterable[GenerationBenchmarkRecord], mode_pair: tuple[str, str], title: str):
    baseline_mode, candidate_mode = mode_pair
    grouped = _group_success_records(records)
    lines = [f'### {title}', '']
    rendered = False

    for dataset_name in sorted(grouped):
        dataset_rows = grouped[dataset_name]
        baseline = dataset_rows.get(baseline_mode)
        candidate = dataset_rows.get(candidate_mode)
        if baseline is None or candidate is None:
            continue
        if baseline.mean_fit_sec is None or candidate.mean_fit_sec is None:
            continue
        improvement = _relative_improvement(baseline.mean_fit_sec, candidate.mean_fit_sec, lower_is_better=True)
        lines.append(
            f'- `{dataset_name}`: средний fit `{baseline_mode}`={baseline.mean_fit_sec:.4f}s, '
            f'`{candidate_mode}`={candidate.mean_fit_sec:.4f}s, улучшение={improvement:+.2f}%.'
        )
        rendered = True

    if not rendered:
        lines.append('- Недостаточно успешных прогонов для сравнения по этой паре режимов.')
    lines.append('')
    return lines


def _render_generation_hypothesis(records: Iterable[GenerationBenchmarkRecord]):
    grouped = _group_success_records(records)
    lines = []
    rendered = False

    for dataset_name in sorted(grouped):
        dataset_rows = grouped[dataset_name]
        baseline = dataset_rows.get('input_cpu')
        candidate = dataset_rows.get('tensor_cpu')
        if baseline is None or candidate is None:
            continue
        if baseline.wall_clock_sec is None or candidate.wall_clock_sec is None:
            continue

        wall_time_delta = _relative_improvement(baseline.wall_clock_sec, candidate.wall_clock_sec, lower_is_better=True)
        diversity_delta = _relative_improvement(
            baseline.unique_pipeline_ratio or 0.0,
            candidate.unique_pipeline_ratio or 0.0,
            lower_is_better=False,
        )
        quality_line = _format_quality_delta(baseline, candidate)
        lines.append(
            f'- `{dataset_name}`: generation wall time {wall_time_delta:+.2f}%, '
            f'unique_pipeline_ratio {diversity_delta:+.2f}%. {quality_line}'
        )
        rendered = True

    if not rendered:
        lines.append('- Для generation-level сравнения пока не хватает парных успешных прогонов.')
    return lines


def _render_dask_hypothesis(records: Iterable[GenerationBenchmarkRecord], capabilities: SuiteCapabilities):
    if not capabilities.dask_available:
        return [f'- Dask недоступен: {capabilities.dask_reason or "не найден пакет distributed/dask"}.']

    grouped = defaultdict(dict)
    for record in records:
        if record.status != 'success' or not record.mode.endswith('gpu_bridge'):
            continue
        grouped[(record.dataset_name, record.mode, record.seed)][record.executor] = record

    lines = []
    rendered = False
    for dataset_mode_seed in sorted(grouped):
        executor_rows = grouped[dataset_mode_seed]
        sequential = executor_rows.get('sequential')
        dask_row = executor_rows.get('dask_experimental')
        if sequential is None or dask_row is None:
            continue
        if sequential.wall_clock_sec is None or dask_row.wall_clock_sec is None:
            continue
        delta = _relative_improvement(sequential.wall_clock_sec, dask_row.wall_clock_sec, lower_is_better=True)
        dataset_name, mode, seed = dataset_mode_seed
        lines.append(
            f'- `{dataset_name}` / `{mode}` / seed={seed}: wall time delta Dask vs sequential = {delta:+.2f}%.'
        )
        rendered = True

    if not rendered:
        lines.append('- Dask режим не запускался параллельно с sequential GPU runs, поэтому сравнение пока пустое.')
    return lines


def _group_success_records(records: Iterable[GenerationBenchmarkRecord]):
    grouped = defaultdict(dict)
    for record in records:
        if record.status != 'success':
            continue
        key = f'{record.dataset_name}::{record.executor}::{record.seed}'
        grouped[key][record.mode] = record
    return grouped


def _relative_improvement(baseline: float, candidate: float, lower_is_better: bool) -> float:
    if baseline == 0:
        return 0.0
    if lower_is_better:
        return ((baseline - candidate) / abs(baseline)) * 100.0
    return ((candidate - baseline) / abs(baseline)) * 100.0


def _format_quality_delta(baseline: GenerationBenchmarkRecord, candidate: GenerationBenchmarkRecord) -> str:
    if baseline.quality_metric_value is None or candidate.quality_metric_value is None:
        return 'Сравнение качества недоступно.'

    lower_is_better = baseline.quality_metric_name.lower() in {'rmse', 'mse', 'mae'}
    delta = _relative_improvement(
        baseline.quality_metric_value,
        candidate.quality_metric_value,
        lower_is_better=lower_is_better,
    )
    return (
        f'Качество по `{baseline.quality_metric_name}`: '
        f'`{baseline.mode}`={baseline.quality_metric_value:.4f}, '
        f'`{candidate.mode}`={candidate.quality_metric_value:.4f}, '
        f'delta={delta:+.2f}%.'
    )


def _ensure_terminal_period(message: str) -> str:
    return message.rstrip('.') + '.'