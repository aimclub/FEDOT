import pytest

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.schemas import (
    validate_optional_preprocessing_data_type,
    validate_optional_service_is_fitted,
)
from fedot.validation.errors import FedotValidationError


def test_validate_optional_service_fitted_state_accepts_consistent_state():
    validate_optional_service_is_fitted(
        has_plan=True,
        has_handlers=True,
    )


@pytest.mark.parametrize(
    'has_plan, has_handlers',
    [
        (False, False),
        (True, False),
        (False, True),
    ],
)
def test_validate_optional_service_fitted_state_rejects_invalid_state(
    has_plan,
    has_handlers,
):
    with pytest.raises(FedotValidationError, match='not fitted yet'):
        validate_optional_service_is_fitted(
            has_plan=has_plan,
            has_handlers=has_handlers,
        )


@pytest.mark.parametrize(
    'data_type',
    [DataTypesEnum.tabular, DataTypesEnum.ts],
)
def test_validate_optional_preprocessing_data_type_accepts_supported(data_type):
    assert validate_optional_preprocessing_data_type(data_type) is data_type


def test_validate_optional_preprocessing_data_type_rejects_unsupported():
    with pytest.raises(FedotValidationError, match='Optional preprocessing is not supported'):
        validate_optional_preprocessing_data_type(DataTypesEnum.image)
