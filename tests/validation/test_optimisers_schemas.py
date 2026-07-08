import pytest
from golem.core.tuning.simultaneous import SimultaneousTuner

from fedot.core.optimisers.schemas import (
    validate_multi_objective_tuner,
    validate_registered_metric,
)
from fedot.validation.errors import FedotValidationError


def test_validate_registered_metric_accepts_known_metric():
    validate_registered_metric('accuracy')


def test_validate_registered_metric_rejects_unknown_metric():
    with pytest.raises(FedotValidationError, match='Incorrect metric'):
        validate_registered_metric('unknown_metric')


def test_validate_multi_objective_tuner_accepts_single_metric():
    validate_multi_objective_tuner(SimultaneousTuner, 1)


def test_validate_multi_objective_tuner_rejects_unsupported_tuner():
    with pytest.raises(FedotValidationError, match='Multi objective tuning'):
        validate_multi_objective_tuner(SimultaneousTuner, 2)
