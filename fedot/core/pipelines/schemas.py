from marshmallow import INCLUDE, Schema, ValidationError, fields, validates

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext


class ComposerRequirementsSchema(Schema):
    class Meta:
        unknown = INCLUDE

    cv_folds = fields.Int(allow_none=True, load_default=None)

    @validates('cv_folds')
    def validate_cv_folds(self, value) -> None:
        if value is not None and value <= 1:
            raise ValidationError(
                'Number of folds for KFold cross validation must be 2 or more.')


def validate_cv_folds(cv_folds, context: ValidationContext = None) -> None:
    load_validated(
        ComposerRequirementsSchema(),
        {'cv_folds': cv_folds},
        context,
        prefix='composer_requirements',
    )


class PipelineFittedStateSchema(Schema):
    class Meta:
        unknown = INCLUDE

    is_fitted = fields.Bool(required=True)

    @validates('is_fitted')
    def validate_is_fitted(self, value: bool) -> None:
        if not value:
            raise ValidationError('Pipeline is not fitted yet')


class PipelineRootNodesSchema(Schema):
    class Meta:
        unknown = INCLUDE

    root_nodes_count = fields.Int(required=True)

    @validates('root_nodes_count')
    def validate_root_nodes_count(self, value: int) -> None:
        if value > 1:
            raise ValidationError('More than 1 root_nodes in pipeline')


def validate_pipeline_is_fitted(is_fitted: bool, context: ValidationContext = None) -> None:
    load_validated(
        PipelineFittedStateSchema(),
        {'is_fitted': is_fitted},
        context,
        prefix='pipeline',
    )


def validate_single_root_node(root_nodes_count: int, context: ValidationContext = None) -> None:
    load_validated(
        PipelineRootNodesSchema(),
        {'root_nodes_count': root_nodes_count},
        context,
        prefix='pipeline',
    )
