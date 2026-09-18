import pytest

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.planner.planner import PreprocessingPlan
from fedot.preprocessing.schemas import (
    validate_optional_preprocessing_data_type,
    validate_optional_service_predict_ready,
)
from fedot.preprocessing.tools.preprocessor_types import (
    ImputationMethodEnum,
    PreprocessingStep,
    PreprocessingStepEnum,
)
from fedot.validation.errors import FedotValidationError


def _plan_with_steps(n_steps: int) -> PreprocessingPlan:
    plan = PreprocessingPlan()
    for _ in range(n_steps):
        plan.add_step(
            PreprocessingStep(
                PreprocessingStepEnum.imputation,
                ImputationMethodEnum.mean,
                [0],
            )
        )
    return plan


def test_validate_optional_service_predict_ready_accepts_empty_plan():
    validate_optional_service_predict_ready(
        plan=PreprocessingPlan(),
        fitted_handlers=[],
    )


def test_validate_optional_service_predict_ready_accepts_matching_memory_handlers():
    validate_optional_service_predict_ready(
        plan=_plan_with_steps(2),
        fitted_handlers=[object(), object()],
    )


def test_validate_optional_service_predict_ready_accepts_matching_cached_handlers():
    validate_optional_service_predict_ready(
        plan=_plan_with_steps(2),
        fitted_handlers=None,
        cached_handler_paths=['a.pkl', 'b.pkl'],
    )


def test_validate_optional_service_predict_ready_rejects_missing_plan():
    with pytest.raises(FedotValidationError, match='not fitted yet'):
        validate_optional_service_predict_ready(
            plan=None,
            fitted_handlers=[],
        )


def test_validate_optional_service_predict_ready_rejects_missing_handlers_for_steps():
    with pytest.raises(
        FedotValidationError,
        match='All required optional preprocessing handlers must be trained',
    ):
        validate_optional_service_predict_ready(
            plan=_plan_with_steps(2),
            fitted_handlers=[],
        )


def test_validate_optional_service_predict_ready_rejects_length_mismatch():
    with pytest.raises(
        FedotValidationError,
        match='All required optional preprocessing handlers must be trained',
    ):
        validate_optional_service_predict_ready(
            plan=_plan_with_steps(2),
            fitted_handlers=[object()],
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
