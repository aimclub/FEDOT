import logging

import pytest

from fedot.validation.context import ValidationContext
from fedot.validation.schemas.prediction_intervals import validate_prediction_intervals_init


def _default_params_dict():
    return {
        'logging_level': 20,
        'n_jobs': -1,
        'show_progress': True,
        'number_mutations': 30,
        'mutations_choice': 'different',
        'mutations_discard_inapropriate_pipelines': True,
        'mutations_keep_percentage': 0.66,
        'mutations_operations': ['lagged'],
        'ql_number_models': 10,
        'ql_tuner_iterations': 10,
        'ql_tuner_minutes': 1,
        'bpq_number_models': 10,
    }


def test_validate_prediction_intervals_init_valid():
    result = validate_prediction_intervals_init(
        horizon=5,
        nominal_error=0.1,
        method='last_generation_ql',
        params_dict=_default_params_dict(),
    )
    assert result['n_jobs'] == -1
    assert result['ql_number_models'] == 10


def test_validate_prediction_intervals_init_invalid_method_recovers_with_default(caplog):
    caplog.set_level(logging.WARNING)
    context = ValidationContext(logger=logging.getLogger('test_prediction_intervals'))

    result = validate_prediction_intervals_init(
        horizon=5,
        nominal_error=0.1,
        method='invalid_method',
        params_dict=_default_params_dict(),
        context=context,
    )
    assert result['logging_level'] == 20
    assert any('method' in record.message for record in caplog.records)


def test_validate_prediction_intervals_init_ql_number_models_max():
    params = _default_params_dict()
    params['ql_number_models'] = 'max'

    result = validate_prediction_intervals_init(
        horizon=None,
        nominal_error=0.1,
        method='best_pipelines_quantiles',
        params_dict=params,
    )
    assert result['ql_number_models'] == 'max'
