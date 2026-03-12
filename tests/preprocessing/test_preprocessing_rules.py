import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.preprocessing_rules import (
    build_preprocessor_merge_plan,
    iter_preprocessed_inputs,
    resolve_main_target_source_name,
    resolve_source_names,
    should_initialize_source_helpers,
)
from fedot.preprocessing.structure import DEFAULT_SOURCE_NAME


def _make_input_data(*, is_main_target=True):
    return InputData(
        idx=np.array([0, 1]),
        features=np.array([[1.0], [2.0]]),
        target=np.array([[0.0], [1.0]]),
        task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.table,
        supplementary_data=SupplementaryData(is_main_target=is_main_target),
    )


def test_resolve_source_names_handles_unimodal_and_multimodal():
    unimodal_plan = resolve_source_names(_make_input_data(), DEFAULT_SOURCE_NAME)
    multimodal_plan = resolve_source_names(
        MultiModalData({'left': _make_input_data(), 'right': _make_input_data(is_main_target=False)}),
        DEFAULT_SOURCE_NAME,
    )

    assert unimodal_plan.source_names == (DEFAULT_SOURCE_NAME,)
    assert multimodal_plan.source_names == ('left', 'right')


def test_resolve_source_names_rejects_unknown_data_type():
    with pytest.raises(ValueError, match='Unknown type of data'):
        resolve_source_names(object(), DEFAULT_SOURCE_NAME)


def test_should_initialize_source_helpers_reflects_existing_state():
    assert should_initialize_source_helpers(False, False) is True
    assert should_initialize_source_helpers(True, False) is True
    assert should_initialize_source_helpers(True, True) is False


def test_resolve_main_target_source_name_prefers_existing_then_detects_main_branch():
    multi_data = MultiModalData({
        'main': _make_input_data(is_main_target=True),
        'side': _make_input_data(is_main_target=False),
    })

    assert resolve_main_target_source_name('preset', multi_data) == 'preset'
    assert resolve_main_target_source_name(None, multi_data) == 'main'


def test_iter_preprocessed_inputs_and_merge_plan_are_deterministic():
    input_data = _make_input_data()
    multi_data = MultiModalData({'main': input_data, 'side': _make_input_data(is_main_target=False)})

    assert iter_preprocessed_inputs(input_data) == (input_data,)
    assert len(iter_preprocessed_inputs(multi_data)) == 2

    auto_plan = build_preprocessor_merge_plan(True, {'enc': 1}, {'imp': 1})
    manual_plan = build_preprocessor_merge_plan(False, {}, {})

    assert auto_plan.take_pipeline_encoders is False
    assert auto_plan.take_pipeline_imputers is False
    assert manual_plan.take_pipeline_encoders is True
    assert manual_plan.take_pipeline_imputers is True
