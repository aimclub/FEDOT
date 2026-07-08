from typing import Any, Dict, Optional, Set, Type

from marshmallow import RAISE, Schema, ValidationError, fields, validates, validates_schema

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext


def build_api_params_keys_schema(allowed_keys: Set[str]) -> Type[Schema]:
    """Build a schema that rejects unknown API parameter keys."""
    field_definitions = {
        key: fields.Raw(required=False, allow_none=True)
        for key in allowed_keys
    }

    meta = type('Meta', (), {'unknown': RAISE})
    return type('ApiParamsKeysSchema', (Schema,), {**field_definitions, 'Meta': meta})


class ProblemSchema(Schema):
    problem = fields.Str(required=True)

    @validates_schema
    def validate_problem(self, data: Dict[str, Any], **kwargs) -> None:
        supported = {'regression', 'classification', 'ts_forecasting'}
        if data.get('problem') not in supported:
            raise ValidationError(
                f'Wrong type name of the given task: {data.get("problem")}',
                field_name='problem',
            )


class TimeoutGenerationsSchema(Schema):
    timeout = fields.Raw(allow_none=True, load_default=None)
    num_of_generations = fields.Int(allow_none=True, load_default=None)

    @validates_schema
    def validate_timeout_generations(self, data: Dict[str, Any], **kwargs) -> None:
        timeout = data.get('timeout')
        num_of_generations = data.get('num_of_generations')

        if timeout in (-1, None):
            if num_of_generations is None:
                raise ValidationError(
                    '"num_of_generations" should be specified if infinite "timeout" is given',
                    field_name='num_of_generations',
                )
            return

        if timeout is not None and timeout <= 0:
            raise ValidationError(
                f'invalid "timeout" value: timeout={timeout}',
                field_name='timeout',
            )


class TensorMetricsExecutionSchema(Schema):
    class Meta:
        unknown = RAISE

    is_pipeline_fitted = fields.Bool(required=True)
    metric_names = fields.Raw(allow_none=True, load_default=None)
    default_metrics = fields.Raw(required=True)
    requested_in_sample = fields.Bool(allow_none=True, load_default=None)
    default_in_sample = fields.Bool(required=True)
    validation_blocks = fields.Int(allow_none=True, load_default=None)
    rounding_order = fields.Int(load_default=3)

    @validates('is_pipeline_fitted')
    def validate_pipeline_is_fitted(self, value: bool) -> None:
        if not value:
            raise ValidationError('Pipeline is not fitted yet')

    @validates('rounding_order')
    def validate_rounding_order(self, value: int) -> None:
        if value < 0:
            raise ValidationError('rounding_order should be non-negative')


class AssumptionFitErrorSchema(Schema):
    class Meta:
        unknown = RAISE

    message = fields.Str(required=True)
    exception = fields.Raw(allow_none=True, load_default=None)


class TensorDataCreationTraceSchema(Schema):
    class Meta:
        unknown = RAISE

    is_predict = fields.Bool(required=True)
    trace_uuid = fields.Str(allow_none=True, load_default=None)

    @validates_schema
    def validate_trace_uuid(self, data: Dict[str, Any], **kwargs) -> None:
        if data['is_predict'] and data.get('trace_uuid') is None:
            raise ValidationError(
                'trace_uuid is required for TensorData creation in predict state.',
                field_name='trace_uuid',
            )


def validate_api_param_keys(
    params: dict,
    allowed_keys,
    context: ValidationContext = None,
) -> None:
    schema_cls = build_api_params_keys_schema(set(allowed_keys))
    load_validated(schema_cls(), params, context)


def validate_problem(problem: str, context: ValidationContext = None) -> None:
    load_validated(ProblemSchema(), {'problem': problem}, context)


def validate_timeout_generations(
    timeout,
    num_of_generations,
    context: ValidationContext = None,
) -> Dict[str, Any]:
    return load_validated(
        TimeoutGenerationsSchema(),
        {'timeout': timeout, 'num_of_generations': num_of_generations},
        context,
    )


def raise_from_assumption_fit_error(
    fit_error: object,
    context: ValidationContext = None,
) -> None:
    validated = load_validated(
        AssumptionFitErrorSchema(),
        {
            'message': getattr(fit_error, 'message', str(fit_error)),
            'exception': getattr(fit_error, 'exception', None),
        },
        context,
        prefix='assumption_fit',
    )
    raise ValueError(validated['message']) from validated.get('exception')


def validate_tensordata_creation_trace(
    is_predict: bool,
    trace_uuid: Optional[str],
    context: ValidationContext = None,
) -> None:
    load_validated(
        TensorDataCreationTraceSchema(),
        {'is_predict': is_predict, 'trace_uuid': trace_uuid},
        context,
        prefix='tensor_data_creation',
    )
