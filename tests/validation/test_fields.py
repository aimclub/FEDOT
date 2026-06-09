import pytest
from marshmallow import Schema, ValidationError

from fedot.validation.fields import SortedUniqueFloatTuple, validate_sorted_unique_ratios


class RatiosSchema(Schema):
    ratios = SortedUniqueFloatTuple()


def test_sorted_unique_float_tuple_accepts_sorted_unique():
    result = RatiosSchema().load({'ratios': [0.2, 0.5, 0.7]})
    assert result['ratios'] == (0.2, 0.5, 0.7)


def test_sorted_unique_float_tuple_rejects_duplicates():
    with pytest.raises(ValidationError, match='duplicates'):
        RatiosSchema().load({'ratios': [0.2, 0.2]})


def test_validate_sorted_unique_ratios_rejects_duplicates():
    with pytest.raises(ValidationError, match='duplicates'):
        validate_sorted_unique_ratios([0.2, 0.2])


def test_validate_sorted_unique_ratios_rejects_unsorted():
    with pytest.raises(ValidationError, match='sorted'):
        validate_sorted_unique_ratios([0.5, 0.2])
