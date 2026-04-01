from .core import ExecutorKind, TensorBenchmarkConfig, TensorBenchmarkSuiteResult
from .datasets import DATASET_REGISTRY, load_dataset, resolve_dataset_specs
from .report import render_markdown_report

DEFAULT_DATASETS = tuple(DATASET_REGISTRY.keys())
DEFAULT_MODES = ('input_cpu', 'tensor_cpu', 'input_gpu_bridge', 'tensor_gpu_bridge')


def build_config(*args, **kwargs):
    from .runner import build_config as _build_config

    return _build_config(*args, **kwargs)


def detect_capabilities(*args, **kwargs):
    from .runner import detect_capabilities as _detect_capabilities

    return _detect_capabilities(*args, **kwargs)


def run_tabular_tensor_suite(*args, **kwargs):
    from .runner import run_tabular_tensor_suite as _run_tabular_tensor_suite

    return _run_tabular_tensor_suite(*args, **kwargs)


__all__ = [
    'DATASET_REGISTRY',
    'DEFAULT_DATASETS',
    'DEFAULT_MODES',
    'ExecutorKind',
    'TensorBenchmarkConfig',
    'TensorBenchmarkSuiteResult',
    'build_config',
    'detect_capabilities',
    'load_dataset',
    'render_markdown_report',
    'resolve_dataset_specs',
    'run_tabular_tensor_suite',
]
