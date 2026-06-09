from typing import Any, Dict, Set, Type

from marshmallow import RAISE, Schema, ValidationError, fields, validates_schema

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
