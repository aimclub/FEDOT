import logging

import pytest

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext
from fedot.validation.errors import FedotInvalidKeysError
from fedot.core.pipelines.ensembling.schemas import ChunkedEnsembleConfigSchema


def _load_chunked_ensemble_config(config, context=None):
    return load_validated(
        ChunkedEnsembleConfigSchema(),
        config,
        context,
        prefix='chunked_ensemble_config',
    )


def test_chunked_ensemble_config_valid():
    """Happy path: explicitly provided values must be returned as-is."""
    loaded = _load_chunked_ensemble_config({
        'validation_size': 0.25,
        'ensemble_method': 'weighted',
    })
    assert loaded['validation_size'] == 0.25
    assert loaded['ensemble_method'] == 'weighted'


def test_chunked_ensemble_config_defaults():
    """Empty dict must fill all fields with schema defaults.

    Desired behavior: when the user passes no config (or an empty dict), every
    field must receive its ``load_default``: ``validation_size=0.2``,
    ``ensemble_method='voting'``, ``batch_size=10000``. This is the default
    ensemble behavior users get without any configuration.
    """
    loaded = _load_chunked_ensemble_config({})
    assert loaded['validation_size'] == 0.2
    assert loaded['ensemble_method'] == 'voting'
    assert loaded['batch_size'] == 10000


def test_chunked_ensemble_config_unknown_keys_raises():
    """Unknown keys must raise FedotInvalidKeysError (RAISE mode).

    Desired behavior: an unrecognized key like 'unknown' is almost certainly a
    typo. The schema must reject it with ``FedotInvalidKeysError`` rather than
    silently ignore it, so users are forced to fix the misspelling.
    """
    with pytest.raises(FedotInvalidKeysError):
        _load_chunked_ensemble_config({'unknown': 1})


def test_chunked_ensemble_config_invalid_method_recovers_with_default(caplog):
    """Invalid ensemble_method triggers default recovery via the shared kernel.

    Desired behavior: 'invalid' is not one of the four valid methods
    (voting/weighted/routed_weighted/gated_weighted), but the field has
    ``load_default='voting'``. The recovery policy must patch in the default,
    emit a WARNING mentioning 'ensemble_method', and the load must succeed with
    the recovered value.
    """
    caplog.set_level(logging.WARNING)
    context = ValidationContext(logger=logging.getLogger('test_chunked_ensemble'))

    loaded = _load_chunked_ensemble_config(
        {'ensemble_method': 'invalid'},
        context=context,
    )
    assert loaded['ensemble_method'] == 'voting'
    assert any('ensemble_method' in record.message for record in caplog.records)
