import pytest

from fedot.preprocessing.schemas import validate_optional_service_is_fitted
from fedot.validation.errors import FedotValidationError


def test_validate_optional_service_fitted_state_accepts_consistent_state():
    validate_optional_service_is_fitted(
        has_plan=True,
        has_handlers=True,
        plan_steps_count=2,
        handlers_count=2,
    )


@pytest.mark.parametrize(
    'has_plan, has_handlers, plan_steps_count, handlers_count, error',
    [
        (False, False, 0, 0, 'not fitted yet'),
        (True, False, 1, 0, 'not fitted yet'),
        (True, True, 2, 1, 'number of handlers'),
    ],
)
def test_validate_optional_service_fitted_state_rejects_invalid_state(
    has_plan,
    has_handlers,
    plan_steps_count,
    handlers_count,
    error,
):
    with pytest.raises(FedotValidationError, match=error):
        validate_optional_service_is_fitted(
            has_plan=has_plan,
            has_handlers=has_handlers,
            plan_steps_count=plan_steps_count,
            handlers_count=handlers_count,
        )
