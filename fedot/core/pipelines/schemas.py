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
