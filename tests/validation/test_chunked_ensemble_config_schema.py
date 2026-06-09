import logging

import pytest

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext
from fedot.validation.errors import FedotInvalidKeysError
from fedot.validation.schemas.chunked_ensemble_config import ChunkedEnsembleConfigSchema


def _load_chunked_ensemble_config(config, context=None):
    return load_validated(
        ChunkedEnsembleConfigSchema(),
        config,
        context,
        prefix='chunked_ensemble_config',
    )


def test_chunked_ensemble_config_valid():
    loaded = _load_chunked_ensemble_config({
        'validation_size': 0.25,
        'ensemble_method': 'weighted',
    })
    assert loaded['validation_size'] == 0.25
    assert loaded['ensemble_method'] == 'weighted'


def test_chunked_ensemble_config_defaults():
    loaded = _load_chunked_ensemble_config({})
    assert loaded['validation_size'] == 0.2
    assert loaded['ensemble_method'] == 'voting'
    assert loaded['batch_size'] == 10000


def test_chunked_ensemble_config_unknown_keys_raises():
    with pytest.raises(FedotInvalidKeysError):
        _load_chunked_ensemble_config({'unknown': 1})


def test_chunked_ensemble_config_invalid_method_recovers_with_default(caplog):
    caplog.set_level(logging.WARNING)
    context = ValidationContext(logger=logging.getLogger('test_chunked_ensemble'))

    loaded = _load_chunked_ensemble_config(
        {'ensemble_method': 'invalid'},
        context=context,
    )
    assert loaded['ensemble_method'] == 'voting'
    assert any('ensemble_method' in record.message for record in caplog.records)
