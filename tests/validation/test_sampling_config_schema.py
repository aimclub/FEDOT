import logging

import pytest

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext
from fedot.validation.errors import FedotInvalidKeysError, FedotValidationError
from fedot.validation.schemas.sampling_config import SamplingConfigSchema


def _load_sampling_config(config, context=None):
    return load_validated(SamplingConfigSchema(), config, context, prefix='sampling_config')


def test_sampling_config_subset_valid():
    loaded = _load_sampling_config({
        'strategy_kind': 'subset',
        'provider': 'sampling_zoo',
    })
    assert loaded['strategy_kind'] == 'subset'
    assert loaded['provider'] == 'sampling_zoo'
    assert loaded['candidate_ratios'] == (0.15, 0.2, 0.3, 0.5, 0.7)


def test_sampling_config_chunking_valid():
    loaded = _load_sampling_config({
        'strategy_kind': 'chunking',
        'provider': 'sampling_zoo',
    })
    assert loaded['strategy_kind'] == 'chunking'
    assert 'candidate_ratios' not in loaded


def test_sampling_config_missing_strategy_kind_raises():
    with pytest.raises(FedotValidationError, match='strategy_kind'):
        _load_sampling_config({'provider': 'sampling_zoo'})


def test_sampling_config_invalid_strategy_kind_raises():
    with pytest.raises(FedotValidationError, match='strategy_kind'):
        _load_sampling_config({'strategy_kind': 'other'})


def test_sampling_config_non_dict_raises():
    with pytest.raises(FedotValidationError, match='dictionary'):
        _load_sampling_config('not-a-dict')


def test_sampling_config_unknown_keys_raises():
    with pytest.raises(FedotInvalidKeysError):
        _load_sampling_config({
            'strategy_kind': 'subset',
            'unknown_key': 1,
        })


def test_sampling_config_invalid_ratio_recovers_with_default(caplog):
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger('test_sampling_config')
    context = ValidationContext(logger=logger)

    loaded = _load_sampling_config({
        'strategy_kind': 'subset',
        'cap_max_timeout_share': 99.0,
    }, context=context)

    assert loaded['cap_max_timeout_share'] == 0.35
    assert any('cap_max_timeout_share' in record.message for record in caplog.records)
