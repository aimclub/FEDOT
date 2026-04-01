from dataclasses import dataclass
from typing import Any, Optional

from fedot.core.data.data import InputData


BENCHMARK_RUNTIME_MODES = (
    'input_cpu',
    'tensor_cpu',
    'input_gpu_bridge',
    'tensor_gpu_bridge',
)


@dataclass(frozen=True)
class BenchmarkRuntimePlan:
    mode: str
    use_tensor_train: bool
    use_tensor_predict: bool
    tensor_backend_name: str
    operation_device: str
    data_mode: str
    use_gpu_bridge: bool


@dataclass(frozen=True)
class RuntimeFoldData:
    input_data: InputData
    tensor_data: Any = None
    runtime_mode: str = 'input_cpu'

    @property
    def use_tensor_runtime(self) -> bool:
        return self.tensor_data is not None


@dataclass(frozen=True)
class RuntimeFitPlan:
    fit_method_name: str
    fit_data: Any
    input_data: InputData
    runtime_mode: str


@dataclass(frozen=True)
class RuntimePredictPlan:
    predict_method_name: str
    predict_data: Any
    reference_input_data: InputData
    runtime_mode: str


@dataclass
class RuntimeEvaluationTelemetry:
    fit_time_sec: float = 0.0
    predict_time_sec: float = 0.0


@dataclass(frozen=True)
class RuntimeEvaluationRecord:
    pipeline_id: str
    runtime_mode: str
    fold_id: int
    n_nodes: int
    n_model_nodes: int
    fit_time_sec: float
    predict_time_sec: float
    objective_values: tuple[float, ...]
    success: bool


def normalize_benchmark_runtime_mode(runtime_mode: Optional[str]) -> str:
    if runtime_mode is None:
        return 'input_cpu'
    normalized_mode = runtime_mode.strip().lower()
    if normalized_mode not in BENCHMARK_RUNTIME_MODES:
        supported = ', '.join(BENCHMARK_RUNTIME_MODES)
        raise ValueError(
            f'Unsupported benchmark runtime mode: {runtime_mode}. Expected one of: {supported}'
        )
    return normalized_mode


def build_benchmark_runtime_plan(runtime_mode: Optional[str],
                                 tensor_backend_name: Optional[str] = None) -> BenchmarkRuntimePlan:
    normalized_mode = normalize_benchmark_runtime_mode(runtime_mode)
    is_tensor_mode = normalized_mode.startswith('tensor_')
    is_gpu_mode = normalized_mode.endswith('gpu_bridge')
    resolved_backend_name = tensor_backend_name or ('gpu' if is_gpu_mode else 'cpu')

    return BenchmarkRuntimePlan(
        mode=normalized_mode,
        use_tensor_train=is_tensor_mode,
        use_tensor_predict=is_tensor_mode,
        tensor_backend_name=resolved_backend_name,
        operation_device='gpu' if is_gpu_mode else 'cpu',
        data_mode='tensor' if is_tensor_mode else 'input',
        use_gpu_bridge=is_gpu_mode,
    )


def ensure_runtime_fold_data(data: Any, runtime_mode: str = 'input_cpu') -> RuntimeFoldData:
    if isinstance(data, RuntimeFoldData):
        return RuntimeFoldData(
            input_data=data.input_data,
            tensor_data=data.tensor_data,
            runtime_mode=normalize_benchmark_runtime_mode(data.runtime_mode),
        )
    if not isinstance(data, InputData):
        raise TypeError(f'Expected InputData or RuntimeFoldData, got {type(data)}')
    return RuntimeFoldData(input_data=data, tensor_data=None, runtime_mode=normalize_benchmark_runtime_mode(runtime_mode))


def build_runtime_fit_plan(train_data: Any) -> RuntimeFitPlan:
    runtime_data = ensure_runtime_fold_data(train_data)
    if runtime_data.use_tensor_runtime:
        return RuntimeFitPlan(
            fit_method_name='fit_tensordata',
            fit_data=runtime_data.tensor_data,
            input_data=runtime_data.input_data,
            runtime_mode=runtime_data.runtime_mode,
        )
    return RuntimeFitPlan(
        fit_method_name='fit',
        fit_data=runtime_data.input_data,
        input_data=runtime_data.input_data,
        runtime_mode=runtime_data.runtime_mode,
    )


def build_runtime_predict_plan(reference_data: Any) -> RuntimePredictPlan:
    runtime_data = ensure_runtime_fold_data(reference_data)
    if runtime_data.use_tensor_runtime:
        return RuntimePredictPlan(
            predict_method_name='predict_tensordata',
            predict_data=runtime_data.tensor_data,
            reference_input_data=runtime_data.input_data,
            runtime_mode=runtime_data.runtime_mode,
        )
    return RuntimePredictPlan(
        predict_method_name='predict',
        predict_data=runtime_data.input_data,
        reference_input_data=runtime_data.input_data,
        runtime_mode=runtime_data.runtime_mode,
    )


def with_tensor_runtime(input_data: InputData, tensor_data: Any, runtime_mode: str) -> RuntimeFoldData:
    return RuntimeFoldData(
        input_data=input_data,
        tensor_data=tensor_data,
        runtime_mode=normalize_benchmark_runtime_mode(runtime_mode),
    )


def with_input_runtime(input_data: InputData, runtime_mode: str) -> RuntimeFoldData:
    return RuntimeFoldData(
        input_data=input_data,
        tensor_data=None,
        runtime_mode=normalize_benchmark_runtime_mode(runtime_mode),
    )
