import logging

import pytest
from marshmallow import Schema, ValidationError

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext
from fedot.validation.errors import FedotInvalidKeysError, FedotValidationError
from fedot.api.sampling_stage.schemas import (
    SamplingConfigSchema,
    SortedUniqueFloatTuple,
    validate_sorted_unique_ratios,
)


def _load_sampling_config(config, context=None):
    return load_validated(SamplingConfigSchema(), config, context, prefix='sampling_config')


def test_sampling_config_subset_valid():
    """Happy path for the 'subset' strategy kind.

    Desired behavior: when ``strategy_kind='subset'``, the discriminated union
    loader routes to ``SamplingSubsetConfigSchema``. All fields must pass
    validation, and fields omitted by the user must be filled with their
    schema defaults (e.g. ``candidate_ratios`` gets the built-in default tuple).
    """
    loaded = _load_sampling_config({
        'strategy_kind': 'subset',
        'provider': 'sampling_zoo',
    })
    assert loaded['strategy_kind'] == 'subset'
    assert loaded['provider'] == 'sampling_zoo'
    assert loaded['candidate_ratios'] == (0.15, 0.2, 0.3, 0.5, 0.7)


def test_sampling_config_chunking_valid():
    """Happy path for the 'chunking' strategy kind.

    Desired behavior: when ``strategy_kind='chunking'``, the loader routes to
    ``SamplingChunkingConfigSchema``. Subset-specific fields (e.g.
    ``candidate_ratios``) must *not* appear in the output since they do not
    belong to the chunking variant.
    """
    loaded = _load_sampling_config({
        'strategy_kind': 'chunking',
        'provider': 'sampling_zoo',
    })
    assert loaded['strategy_kind'] == 'chunking'
    assert 'candidate_ratios' not in loaded


def test_sampling_config_missing_strategy_kind_raises():
    """The discriminator field is required and cannot be defaulted.

    Desired behavior: if ``strategy_kind`` is missing, the discriminated-union
    loader cannot determine which sub-schema to use, so it must raise
    ``FedotValidationError`` immediately (not attempt recovery).
    """
    with pytest.raises(FedotValidationError, match='strategy_kind'):
        _load_sampling_config({'provider': 'sampling_zoo'})


def test_sampling_config_invalid_strategy_kind_raises():
    """An unrecognized strategy kind is rejected at the discriminator level.

    Desired behavior: 'other' is not one of the valid dispatch keys
    ('subset', 'chunking'), so the loader must raise rather than silently
    falling through to either sub-schema.
    """
    with pytest.raises(FedotValidationError, match='strategy_kind'):
        _load_sampling_config({'strategy_kind': 'other'})


def test_sampling_config_non_dict_raises():
    """Non-dict input is caught before dispatch.

    Desired behavior: passing a string instead of a dict must be caught by
    ``SamplingConfigSchema.load`` and converted to ``FedotValidationError``.
    """
    with pytest.raises(FedotValidationError, match='dictionary'):
        _load_sampling_config('not-a-dict')


def test_sampling_config_unknown_keys_raises():
    """Unknown keys raise FedotInvalidKeysError, same as any RAISE-mode schema.

    Desired behavior: an extra key that doesn't belong to either the base or
    the subset sub-schema must surface as ``FedotInvalidKeysError``, not be
    silently dropped. This prevents typos in config from going unnoticed.
    """
    with pytest.raises(FedotInvalidKeysError):
        _load_sampling_config({
            'strategy_kind': 'subset',
            'unknown_key': 1,
        })


def test_sampling_config_invalid_ratio_recovers_with_default(caplog):
    """The default-recovery policy applies within the discriminated union.

    Desired behavior: ``cap_max_timeout_share=99.0`` violates the
    ``(0, 1]`` constraint. Because the field has a ``load_default=0.35``,
    the recovery machinery must patch in 0.35, emit a WARNING through the
    context logger mentioning the field name, and the overall load must
    succeed with the recovered value.
    """
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger('test_sampling_config')
    context = ValidationContext(logger=logger)

    loaded = _load_sampling_config({
        'strategy_kind': 'subset',
        'cap_max_timeout_share': 99.0,
    }, context=context)

    assert loaded['cap_max_timeout_share'] == 0.35
    assert any('cap_max_timeout_share' in record.message for record in caplog.records)


# ---- SortedUniqueFloatTuple field-type tests ----
# (Relocated from tests/validation/test_fields.py — this field is
#  sampling-stage-specific, not shared with other modules.)


class _RatiosSchema(Schema):
    ratios = SortedUniqueFloatTuple()


def test_sorted_unique_float_tuple_accepts_sorted_unique():
    """A sorted, unique, in-range list must deserialize to a tuple of floats.

    Desired behavior: ``[0.2, 0.5, 0.7]`` is valid — all values are in
    ``(0, 1]``, unique, and in ascending order. The result must be a
    ``tuple`` (not a list) so downstream consumers get an immutable value.
    """
    result = _RatiosSchema().load({'ratios': [0.2, 0.5, 0.7]})
    assert result['ratios'] == (0.2, 0.5, 0.7)


def test_sorted_unique_float_tuple_rejects_duplicates():
    """Duplicate ratios must be rejected with a clear 'duplicates' message.

    Desired behavior: ``[0.2, 0.2]`` contains a duplicate, which would make
    the sampling strategy evaluate the same subspace twice. The field must
    raise ``ValidationError`` (marshmallow-level, caught and re-raised as
    ``FedotValidationError`` by the boundary) with 'duplicates' in the
    message so the cause is obvious.
    """
    with pytest.raises(ValidationError, match='duplicates'):
        _RatiosSchema().load({'ratios': [0.2, 0.2]})


def test_validate_sorted_unique_ratios_rejects_duplicates():
    """The standalone validator also catches duplicates (used internally by
    ``SortedUniqueFloatTuple._deserialize``).

    Desired behavior: same contract as the field-level test, but exercised on
    the bare function. Ensures the function is correct independently of the
    marshmallow deserialization wrapper.
    """
    with pytest.raises(ValidationError, match='duplicates'):
        validate_sorted_unique_ratios([0.2, 0.2])


def test_validate_sorted_unique_ratios_rejects_unsorted():
    """Unsorted ratios must be rejected with a 'sorted' message.

    Desired behavior: ``[0.5, 0.2]`` is valid in terms of range and uniqueness
    but is not in ascending order. The sampling stage relies on sorted ratios to
    define a monotonic search space, so out-of-order input must be rejected
    rather than silently re-sorted (which could mask a user mistake).
    """
    with pytest.raises(ValidationError, match='sorted'):
        validate_sorted_unique_ratios([0.5, 0.2])
