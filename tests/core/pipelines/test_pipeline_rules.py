import pytest

from fedot.core.pipelines.pipeline_rules import (
    OutputModeEnum,
    build_pipeline_postprocess_plan,
    normalize_output_mode,
)
from fedot.core.pipelines.schemas import validate_pipeline_output_mode
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.validation.errors import FedotValidationError


def test_normalize_output_mode_accepts_enum_and_string():
    assert normalize_output_mode(OutputModeEnum.AUTO) is OutputModeEnum.AUTO
    assert normalize_output_mode('decoded') is OutputModeEnum.DECODED


def test_validate_pipeline_output_mode_rejects_unknown_value():
    with pytest.raises(FedotValidationError, match='Invalid output mode'):
        validate_pipeline_output_mode('labels')


def test_build_pipeline_postprocess_plan_for_auto_modes():
    classification_plan = build_pipeline_postprocess_plan(
        'auto', TaskTypesEnum.classification)
    ts_plan = build_pipeline_postprocess_plan(
        OutputModeEnum.AUTO, TaskTypesEnum.ts_forecasting)

    assert classification_plan.should_restore_inverse_target_encoding is True
    assert classification_plan.should_flatten_prediction is False
    assert ts_plan.should_flatten_prediction is True
    assert ts_plan.should_restore_inverse_target_encoding is False


def test_build_pipeline_postprocess_plan_for_explicit_modes():
    raw_plan = build_pipeline_postprocess_plan('raw', TaskTypesEnum.classification)
    decoded_plan = build_pipeline_postprocess_plan(
        'decoded', TaskTypesEnum.classification)
    flat_plan = build_pipeline_postprocess_plan(
        'flattened', TaskTypesEnum.ts_forecasting)

    assert raw_plan == build_pipeline_postprocess_plan(
        OutputModeEnum.RAW, TaskTypesEnum.regression)
    assert decoded_plan.should_restore_inverse_target_encoding is True
    assert decoded_plan.should_flatten_prediction is False
    assert flat_plan.should_flatten_prediction is True
    assert flat_plan.should_restore_inverse_target_encoding is False
