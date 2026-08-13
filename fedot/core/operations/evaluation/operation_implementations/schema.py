from typing import Any, Mapping, Optional

from marshmallow import INCLUDE, Schema, ValidationError, fields, pre_load, validates

from fedot.core.operations.evaluation.operation_implementations.rules import (
    default_pca_n_components,
    has_enough_pca_fit_samples,
    is_valid_pca_n_components,
    pca_fit_samples_error_message,
    pca_n_components_error_message,
)
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
            payload['n_components'] = default_pca_n_components()
        return payload

    @validates('n_components')
    def validate_n_components(self, value: Any) -> None:
        if not is_valid_pca_n_components(value):
            raise ValidationError(pca_n_components_error_message(value))


class PCAFitSamplesSchema(Schema):
    """Runtime check: enough finite samples remain for PCA fit after NaN drop."""

    class Meta:
        unknown = INCLUDE

    n_samples = fields.Raw(required=True)

    @validates('n_samples')
    def validate_n_samples(self, value: Any) -> None:
        if not has_enough_pca_fit_samples(value):
            raise ValidationError(pca_fit_samples_error_message(value))


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


def validate_pca_fit_samples(
    n_samples: Any,
    context: ValidationContext = None,
) -> int:
    """Ensure PCA fit has enough finite samples after dropping NaN rows."""
    validated = load_validated(
        PCAFitSamplesSchema(),
        {'n_samples': n_samples},
        context,
        prefix='pca',
    )
    return validated['n_samples']
