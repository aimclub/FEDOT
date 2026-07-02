from pathlib import Path

from marshmallow import RAISE, Schema, ValidationError, fields, validates

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext


class TabularFilePathSchema(Schema):
    class Meta:
        unknown = RAISE

    file_path = fields.Str(required=True)

    @validates('file_path')
    def validate_file_path(self, value: str) -> None:
        normalized_path = str(value)
        supported_suffixes = ('.csv', '.tsv')

        if not normalized_path.lower().endswith(supported_suffixes):
            raise ValidationError(f'Unsupported tabular file format: {normalized_path}')

        if not Path(normalized_path).is_file():
            raise ValidationError(f'File {normalized_path} does not exist')


def validate_tabular_file_path(
    file_path: str,
    context: ValidationContext = None,
) -> str:
    result = load_validated(
        TabularFilePathSchema(),
        {'file_path': file_path},
        context,
        prefix='tensor_data',
    )
    return result['file_path']
