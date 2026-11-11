import pytest

from fedot.core.data.merge.rules import plan_tensor_merge_input


@pytest.mark.unit
def test_tensor_merge_uses_prediction_only_for_model_output():
    assert plan_tensor_merge_input(
        has_prediction=True, parent_is_model=True).use_predict is True
    assert plan_tensor_merge_input(
        has_prediction=True, parent_is_model=False).use_predict is False
    assert plan_tensor_merge_input(has_prediction=False).use_predict is False


@pytest.mark.unit
def test_tensor_merge_rejects_model_output_without_prediction():
    with pytest.raises(ValueError, match='must provide predict'):
        plan_tensor_merge_input(has_prediction=False, parent_is_model=True)
