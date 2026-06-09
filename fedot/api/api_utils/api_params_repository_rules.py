from dataclasses import asdict
from typing import Any, Callable, Dict

from fedot.core.constants import AUTO_PRESET_NAME
from fedot.core.repository.tasks import TaskTypesEnum


def default_cv_folds_for_task(task_type: TaskTypesEnum) -> int:
    if task_type in (TaskTypesEnum.classification, TaskTypesEnum.regression):
        return 5
    if task_type == TaskTypesEnum.ts_forecasting:
        return 3
    raise ValueError(f'Unsupported task type for default params: {task_type}')


def build_default_api_params(task_type: TaskTypesEnum, default_data_dir: str) -> dict:
    return dict(
        parallelization_mode='populational',
        show_progress=True,
        max_depth=6,
        max_arity=3,
        pop_size=20,
        num_of_generations=None,
        keep_n_best=1,
        available_operations=None,
        metric=None,
        cv_folds=default_cv_folds_for_task(task_type),
        genetic_scheme=None,
        early_stopping_iterations=None,
        early_stopping_timeout=10,
        optimizer=None,
        collect_intermediate_metric=False,
        max_pipeline_fit_time=None,
        initial_assumption=None,
        preset=AUTO_PRESET_NAME,
        use_operations_cache=True,
        use_preprocessing_cache=True,
        use_predictions_cache=True,
        use_stats=False,
        use_input_preprocessing=True,
        use_auto_preprocessing=False,
        use_meta_rules=False,
        cache_dir=default_data_dir,
        keep_history=True,
        history_dir=default_data_dir,
        with_tuning=True,
        seed=None,
        sampling_config=None,
        chunked_ensemble_config=None,
        tensor_data_config=None,
    )


def validate_api_param_keys(params: dict, allowed_keys) -> None:
    invalid_keys = params.keys() - set(allowed_keys)
    if invalid_keys:
        raise KeyError(f'Invalid key parameters {invalid_keys}')


def normalize_sampling_config(config: Any, validator: Callable[[Any], Any]):
    validated_sampling_config = validator(config)
    return asdict(validated_sampling_config) if validated_sampling_config else None


def normalize_chunked_ensemble_config(config: Any, validator: Callable[[Any], Any]):
    if config is None:
        return None
    return validator(config).to_dict()


def normalize_tensor_data_config(config: Any, validator: Callable[[Any], Any]):
    if config is None:
        return None
    return validator(config)


def apply_default_params(params: Dict[str, Any],
                         default_params: Dict[str, Any],
                         sampling_validator: Callable[[Any], Any],
                         chunked_ensemble_validator: Callable[[Any], Any],
                         tensor_data_validator: Callable[[Any], Any]) -> Dict[str, Any]:
    validate_api_param_keys(params, default_params.keys())

    normalized_params = dict(params)
    if 'sampling_config' in normalized_params:
        normalized_params['sampling_config'] = normalize_sampling_config(
            normalized_params['sampling_config'],
            sampling_validator,
        )
    if 'chunked_ensemble_config' in normalized_params:
        normalized_params['chunked_ensemble_config'] = normalize_chunked_ensemble_config(
            normalized_params['chunked_ensemble_config'],
            chunked_ensemble_validator,
        )
    if 'tensor_data_config' in normalized_params:
        normalized_params['tensor_data_config'] = normalize_tensor_data_config(
            normalized_params['tensor_data_config'],
            tensor_data_validator,
        )

    for key, value in default_params.items():
        if key not in normalized_params and value is not None:
            normalized_params[key] = value

    return normalized_params
