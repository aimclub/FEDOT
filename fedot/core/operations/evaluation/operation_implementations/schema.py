from typing import Any, Mapping, Optional

from marshmallow import INCLUDE, Schema, ValidationError, fields, pre_load, validates

from fedot.core.operations.evaluation.operation_implementations.rules import (
    PCA_OPERATION_TYPE,
    TRUNCATED_SVD_OPERATION_TYPE,
    decomposition_fit_samples_error_message,
    has_enough_decomposition_fit_samples,
    is_valid_pca_n_components,
    is_valid_truncated_svd_n_components,
    pca_n_components_error_message,
    truncated_svd_n_components_error_message,
)
from fedot.core.operations.operation_parameters import get_default_params
from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext


class PCAParamsSchema(Schema):
    """Hyperparameters for Tensor PCA pipeline node.

    Missing / ``None`` ``n_components`` is filled from repository defaults in
    ``@pre_load``. Explicitly invalid values fail (no soft recovery via
    ``load_default``).
    """

    class Meta:
        unknown = INCLUDE

    n_components = fields.Raw(required=True)

    @pre_load
    def fill_missing_n_components(self, data, **kwargs):
        payload = dict(data or {})
        if payload.get('n_components') is None:
            payload['n_components'] = get_default_params(PCA_OPERATION_TYPE).get('n_components')
        return payload

    @validates('n_components')
    def validate_n_components(self, value: Any) -> None:
        if not is_valid_pca_n_components(value):
            raise ValidationError(pca_n_components_error_message(value))


class TruncatedSVDParamsSchema(Schema):
    """Hyperparameters for TruncatedSVD: ``n_components`` is a positive int only."""

    class Meta:
        unknown = INCLUDE

    n_components = fields.Raw(required=True)
    n_iter = fields.Integer(load_default=5)
    n_oversamples = fields.Integer(load_default=10)

    @pre_load
    def fill_missing_n_components(self, data, **kwargs):
        payload = dict(data or {})
        if payload.get('n_components') is None:
            payload['n_components'] = get_default_params(TRUNCATED_SVD_OPERATION_TYPE).get(
                'n_components'
            )
        return payload

    @validates('n_components')
    def validate_n_components(self, value: Any) -> None:
        if not is_valid_truncated_svd_n_components(value):
            raise ValidationError(truncated_svd_n_components_error_message(value))

    @validates('n_iter')
    def validate_n_iter(self, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValidationError(f'TruncatedSVD n_iter must be a non-negative int, got {value!r}')

    @validates('n_oversamples')
    def validate_n_oversamples(self, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValidationError(
                f'TruncatedSVD n_oversamples must be a non-negative int, got {value!r}'
            )


class DecompositionFitSamplesSchema(Schema):
    """Runtime check: enough finite samples remain after NaN drop."""

    class Meta:
        unknown = INCLUDE

    n_samples = fields.Raw(required=True)
    op_name = fields.String(load_default='decomposition')

    @validates('n_samples')
    def validate_n_samples(self, value: Any) -> None:
        if not has_enough_decomposition_fit_samples(value):
            op_name = self.context.get('op_name', 'decomposition')
            raise ValidationError(decomposition_fit_samples_error_message(value, op_name=op_name))


def validate_pca_params(
    params: Optional[Mapping[str, Any]] = None,
    context: ValidationContext = None,
) -> dict:
    """Validate / complete PCA params via marshmallow (defaults from repository)."""
    return load_validated(
        PCAParamsSchema(),
        dict(params or {}),
        context,
        prefix='pca',
    )


def validate_truncated_svd_params(
    params: Optional[Mapping[str, Any]] = None,
    context: ValidationContext = None,
) -> dict:
    """Validate / complete TruncatedSVD params (int ``n_components`` only)."""
    return load_validated(
        TruncatedSVDParamsSchema(),
        dict(params or {}),
        context,
        prefix='truncated_svd',
    )


def validate_decomposition_fit_samples(
    n_samples: Any,
    *,
    op_name: str = 'decomposition',
    context: ValidationContext = None,
) -> int:
    """Ensure a linear decomposition fit has enough finite samples."""
    schema = DecompositionFitSamplesSchema()
    schema.context['op_name'] = op_name
    validated = load_validated(
        schema,
        {'n_samples': n_samples, 'op_name': op_name},
        context,
        prefix=op_name,
    )
    return validated['n_samples']
