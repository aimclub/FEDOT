from typing import Any, Dict, Mapping

from marshmallow import INCLUDE, Schema, ValidationError, fields, validates

from fedot.core.operations.rules import (
    OPTIONAL_STRATEGY_STAGE_PARAM_KEYS,
    allowed_optional_strategy_methods,
    is_optional_auto_method,
    is_optional_none_method,
    normalize_optional_method_name,
    resolve_optional_imputation_method,
    resolve_optional_scaling_method,
    supported_optional_imputation_method_names,
    supported_optional_scaling_method_names,
    supported_optional_strategy_steps,
)
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.tools.preprocessor_types import (
    ImputationMethodEnum,
    PreprocessingStepEnum,
    ScalingMethodEnum,
)
from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext
from fedot.validation.errors import FedotInvalidKeysError, FedotValidationError

SUPPORTED_CLASSIFICATION_OUTPUT_MODES = ('labels', 'probs', 'full_probs', 'default')
_SKIP_STAGE = object()


class ClassificationOutputModeSchema(Schema):
    class Meta:
        unknown = INCLUDE

    output_mode = fields.Raw(required=True)

    @validates('output_mode')
    def validate_output_mode(self, value: Any) -> None:
        if value is False or value in SUPPORTED_CLASSIFICATION_OUTPUT_MODES:
            return
        raise ValidationError(f'Output model {value} is not supported')


def validate_classification_output_mode(
    output_mode: Any,
    context: ValidationContext = None,
) -> Any:
    validated = load_validated(
        ClassificationOutputModeSchema(),
        {'output_mode': output_mode},
        context,
        prefix='classification',
    )
    return validated['output_mode']


class OptionalImputationMethodSchema(Schema):
    class Meta:
        unknown = INCLUDE

    imputation_method = fields.Raw(required=True)

    @validates('imputation_method')
    def validate_imputation_method(self, value: Any) -> None:
        method_name = normalize_optional_method_name(value)
        if method_name not in supported_optional_imputation_method_names():
            raise ValidationError(
                f'Unsupported imputation_method for optional preprocessing: {value!r}'
            )


class OptionalScalingMethodSchema(Schema):
    class Meta:
        unknown = INCLUDE

    scaling_method = fields.Raw(required=True)

    @validates('scaling_method')
    def validate_scaling_method(self, value: Any) -> None:
        method_name = normalize_optional_method_name(value)
        if method_name not in supported_optional_scaling_method_names():
            raise ValidationError(
                f'Unsupported scaling_method for optional preprocessing: {value!r}'
            )


def validate_optional_imputation_method(
    imputation_method: Any,
    context: ValidationContext = None,
) -> Any:
    validated = load_validated(
        OptionalImputationMethodSchema(),
        {'imputation_method': imputation_method},
        context,
        prefix='optional_preprocessing',
    )
    return validated['imputation_method']


def validate_optional_scaling_method(
    scaling_method: Any,
    context: ValidationContext = None,
) -> Any:
    validated = load_validated(
        OptionalScalingMethodSchema(),
        {'scaling_method': scaling_method},
        context,
        prefix='optional_preprocessing',
    )
    return validated['scaling_method']


def _parse_optional_strategy_step(raw_step: Any) -> PreprocessingStepEnum:
    if isinstance(raw_step, PreprocessingStepEnum):
        step = raw_step
    else:
        try:
            step = PreprocessingStepEnum(raw_step)
        except (TypeError, ValueError) as exc:
            raise FedotValidationError(
                f'Unknown optional preprocessing step: {raw_step!r}',
                field_name='strategy',
            ) from exc

    if step not in supported_optional_strategy_steps():
        raise FedotValidationError(
            f'Unknown optional preprocessing step: {raw_step!r}',
            field_name='strategy',
        )
    return step


def _ensure_optional_strategy_method_allowed(step: PreprocessingStepEnum, method: Any) -> None:
    allowed = allowed_optional_strategy_methods(step)
    if allowed is None:
        return
    if method in allowed:
        return
    method_name = normalize_optional_method_name(method)
    if method_name in allowed:
        return
    raise FedotValidationError(
        f'Unsupported method for optional preprocessing step {step.value!r}: {method!r}',
        field_name='strategy',
    )


def _normalize_optional_method_only_config(
    step: PreprocessingStepEnum,
    method: Any,
    data_type: DataTypesEnum,
) -> Any:
    _ensure_optional_strategy_method_allowed(step, method)
    if is_optional_none_method(method):
        return _SKIP_STAGE
    if is_optional_auto_method(method):
        return None

    if step == PreprocessingStepEnum.imputation:
        if isinstance(method, ImputationMethodEnum):
            return method
        return resolve_optional_imputation_method(method, data_type)

    if step == PreprocessingStepEnum.scaling:
        if isinstance(method, ScalingMethodEnum):
            return method
        return resolve_optional_scaling_method(method, data_type)

    return method


def _normalize_optional_stage_params(
    step: PreprocessingStepEnum,
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    unknown = set(params) - OPTIONAL_STRATEGY_STAGE_PARAM_KEYS
    if unknown:
        raise FedotInvalidKeysError(unknown, prefix=f'strategy.{step.value}')

    if 'method' not in params:
        raise FedotValidationError(
            f'Optional preprocessing stage {step.value!r} params require "method"',
            field_name='strategy',
        )
    if 'features_idx' not in params:
        raise FedotValidationError(
            f'Optional preprocessing stage {step.value!r} params require "features_idx"',
            field_name='strategy',
        )

    _ensure_optional_strategy_method_allowed(step, params['method'])
    return dict(params)


def _normalize_optional_stage_config(
    step: PreprocessingStepEnum,
    config: Any,
    data_type: DataTypesEnum,
) -> Any:
    if config is None or isinstance(config, (str, bool)) or hasattr(config, 'value'):
        return _normalize_optional_method_only_config(step, config, data_type)

    if isinstance(config, Mapping):
        return [_normalize_optional_stage_params(step, config)]

    if isinstance(config, list):
        if not config:
            raise FedotValidationError(
                f'Optional preprocessing stage {step.value!r} params list must be non-empty',
                field_name='strategy',
            )
        normalized_items = []
        for item in config:
            if not isinstance(item, Mapping):
                raise FedotValidationError(
                    f'Optional preprocessing stage {step.value!r} params must be mappings',
                    field_name='strategy',
                )
            normalized_items.append(_normalize_optional_stage_params(step, item))
        return normalized_items

    raise FedotValidationError(
        f'Unsupported optional preprocessing stage config for {step.value!r}: {config!r}',
        field_name='strategy',
    )


def validate_optional_strategy_mapping(
    strategy: Any,
    data_type: DataTypesEnum,
    context: ValidationContext = None,
) -> Dict[PreprocessingStepEnum, Any]:
    """Validate and normalize an optional-preprocessing strategy mapping fail-fast."""
    del context  # reserved for shared validation context wiring
    if not isinstance(strategy, Mapping):
        raise FedotValidationError(
            f'Optional preprocessing strategy must be a mapping, got {type(strategy)!r}',
            field_name='strategy',
        )

    normalized: Dict[PreprocessingStepEnum, Any] = {}
    for raw_step, raw_config in strategy.items():
        step = _parse_optional_strategy_step(raw_step)
        stage_config = _normalize_optional_stage_config(step, raw_config, data_type)
        if stage_config is _SKIP_STAGE:
            continue
        normalized[step] = stage_config
    return normalized
