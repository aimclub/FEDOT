from dataclasses import dataclass
from typing import Optional

from fedot.industrial.core.repository.constanst_repository import (
    FEDOT_TUNER_STRATEGY,
    FEDOT_TUNING_METRICS,
)


FIT_METHOD_NAMES = {
    'federated_automl': '_federated_strategy',
    'kernel_automl': '_kernel_strategy',
    'forecasting_assumptions': '_forecasting_strategy',
    'forecasting_exogenous': '_forecasting_exogenous_strategy',
    'lora_strategy': '_lora_strategy',
    'sampling_strategy': '_sampling_strategy',
}

PREDICT_METHOD_NAMES = {
    'federated_automl': '_federated_predict',
    'kernel_automl': '_kernel_predict',
    'forecasting_assumptions': '_forecasting_predict',
    'forecasting_exogenous': '_forecasting_predict',
    'lora_strategy': '_lora_predict',
    'sampling_strategy': '_sampling_predict',
}


@dataclass(frozen=True)
class IndustrialStrategyDispatchPlan:
    strategy_name: str
    fit_method_name: str
    predict_method_name: str


@dataclass(frozen=True)
class FederatedRuntimePlan:
    use_raf: bool
    raf_workers: Optional[int]
    batch_size: Optional[int]
    timeout: Optional[float]


@dataclass(frozen=True)
class SamplingPredictPlan:
    labels_output: bool
    use_cur_feature_space: bool


@dataclass(frozen=True)
class IndustrialKernelFinetunePlan:
    normalized_tuning_params: dict


@dataclass(frozen=True)
class IndustrialSamplingIterationPlan:
    sampling_rate: float
    result_key: str


def resolve_industrial_strategy_dispatch(strategy_name: str) -> IndustrialStrategyDispatchPlan:
    if strategy_name not in FIT_METHOD_NAMES or strategy_name not in PREDICT_METHOD_NAMES:
        raise ValueError(f'Unsupported industrial strategy: {strategy_name}')
    return IndustrialStrategyDispatchPlan(
        strategy_name=strategy_name,
        fit_method_name=FIT_METHOD_NAMES[strategy_name],
        predict_method_name=PREDICT_METHOD_NAMES[strategy_name],
    )


def build_federated_runtime_plan(n_samples: int,
                                 batch_size_threshold: int,
                                 requested_workers: Optional[int],
                                 timeout: float,
                                 timeout_partition: int,
                                 default_workers: int) -> FederatedRuntimePlan:
    use_raf = n_samples > batch_size_threshold
    if not use_raf:
        return FederatedRuntimePlan(
            use_raf=False,
            raf_workers=None,
            batch_size=None,
            timeout=timeout,
        )

    raf_workers = requested_workers if requested_workers is not None else default_workers
    min_timeout = 0.5
    selected_timeout = round(timeout / timeout_partition)
    adjusted_timeout = max(min_timeout, selected_timeout)
    batch_size = round(n_samples / raf_workers)
    return FederatedRuntimePlan(
        use_raf=True,
        raf_workers=raf_workers,
        batch_size=batch_size,
        timeout=adjusted_timeout,
    )


def build_sampling_predict_plan(mode: str, sampling_algorithm: str) -> SamplingPredictPlan:
    return SamplingPredictPlan(
        labels_output=mode in ['labels', 'default'],
        use_cur_feature_space=sampling_algorithm == 'CUR',
    )


def build_industrial_kernel_finetune_plan(problem: str, tuning_params: Optional[dict]) -> IndustrialKernelFinetunePlan:
    normalized_tuning_params = dict(tuning_params or {})
    normalized_tuning_params['metric'] = FEDOT_TUNING_METRICS[problem]
    normalized_tuning_params['tuner'] = FEDOT_TUNER_STRATEGY['simultaneous']
    return IndustrialKernelFinetunePlan(normalized_tuning_params=normalized_tuning_params)


def build_sampling_iteration_plans(sampling_algorithm: str, sampling_range) -> list[IndustrialSamplingIterationPlan]:
    return [
        IndustrialSamplingIterationPlan(
            sampling_rate=sampling_rate,
            result_key=f'{sampling_algorithm}_sampling_rate_{sampling_rate}',
        )
        for sampling_rate in sampling_range
    ]
