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


def test_validate_optional_service_fitted_state_accepts_cached_handlers_without_memory():
    validate_optional_service_is_fitted(
        has_plan=True,
        has_handlers=False,
        has_cached_handlers=True,
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


def test_validate_optional_preprocessing_data_type_accepts_tabular():
    assert validate_optional_preprocessing_data_type(DataTypesEnum.tabular) is DataTypesEnum.tabular


def test_validate_optional_preprocessing_data_type_accepts_ts_after_industrial_spi():
    # Importing OptionalTSService registers TS optional runtime via SPI.
    from fedot.industrial.core.architecture.preprocessing.ts_optional_service import (  # noqa: F401
        OptionalTSService,
    )

    assert validate_optional_preprocessing_data_type(DataTypesEnum.ts) is DataTypesEnum.ts


def test_validate_optional_preprocessing_data_type_rejects_unsupported():
    with pytest.raises(FedotValidationError, match='Optional preprocessing is not supported'):
        validate_optional_preprocessing_data_type(DataTypesEnum.image)
