from typing import Any, Mapping, Optional

from marshmallow import (
    INCLUDE,
    Schema,
    ValidationError,
    fields,
    post_load,
    pre_load,
    validate,
    validates,
    validates_schema,
)

from fedot.core.operations.evaluation.operation_implementations.rules import (
    PCA_OPERATION_TYPE,
    TRUNCATED_SVD_OPERATION_TYPE,
    SpectrumNComponentsMethod,
    broken_stick_n_error_message,
    decomposition_fit_samples_error_message,
    has_enough_decomposition_fit_samples,
    is_valid_pca_n_components,
    is_valid_truncated_svd_n_components,
    pca_n_components_error_message,
    spectrum_input_error_message,
    spectrum_method_error_message,
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
    """Hyperparameters for TruncatedSVD.

    ``n_components``: positive int, float feature-fraction in ``(0, 1]``,
    ``auto``, ``elbow``, or ``broken_stick``.
    """

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


class BrokenStickNSchema(Schema):
    """Validate broken-stick length ``n`` (number of spectrum pieces)."""

    n = fields.Raw(required=True)

    @validates('n')
    def validate_n(self, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValidationError(broken_stick_n_error_message(value))


class SpectrumRankSelectionSchema(Schema):
    """Validate spectrum rank-selection method and required scree inputs."""

    method = fields.Raw(required=True)
    singular_values = fields.Raw(load_default=None, allow_none=True)
    proportions = fields.Raw(load_default=None, allow_none=True)
    max_components = fields.Integer(required=True, validate=validate.Range(min=1))

    @validates('method')
    def validate_method(self, value: Any) -> None:
        if isinstance(value, SpectrumNComponentsMethod):
            return
        if isinstance(value, str):
            try:
                SpectrumNComponentsMethod(value)
                return
            except ValueError:
                pass
        raise ValidationError(spectrum_method_error_message(value))

    @validates_schema
    def require_spectrum_input(self, data, **kwargs):
        if data.get('singular_values') is None and data.get('proportions') is None:
            raise ValidationError(spectrum_input_error_message(data.get('method')))

    @post_load
    def coerce_method(self, data, **kwargs):
        method = data['method']
        if not isinstance(method, SpectrumNComponentsMethod):
            data['method'] = SpectrumNComponentsMethod(method)
        return data


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
    """Validate / complete TruncatedSVD params (int / fraction / auto / spectrum)."""
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


def validate_broken_stick_n(
    n: Any,
    context: ValidationContext = None,
) -> int:
    """Ensure broken-stick length is a positive int."""
    return load_validated(
        BrokenStickNSchema(),
        {'n': n},
        context,
        prefix='broken_stick',
    )['n']


def validate_spectrum_rank_selection(
    method: Any,
    *,
    singular_values: Any = None,
    proportions: Any = None,
    max_components: int,
    context: ValidationContext = None,
) -> dict:
    """Validate spectrum method + presence of scree inputs; coerce method to enum."""
    return load_validated(
        SpectrumRankSelectionSchema(),
        {
            'method': method,
            'singular_values': singular_values,
            'proportions': proportions,
            'max_components': max_components,
        },
        context,
        prefix='spectrum_n_components',
    )
