import pytest

from fedot.api.api_utils.api_service_rules import (
    build_tensordata_fit_plan,
    validate_tensordata_auto_composition,
)
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum


def test_service_rules_build_tensordata_fit_plan_for_auto_composition():
    plan = build_tensordata_fit_plan('auto')

    assert plan.use_auto_composition is True
    assert plan.fit_method_name is None


def test_service_rules_validate_tensordata_auto_composition_rejects_unsupported_shapes():
    with pytest.raises(ValueError, match='classification and regression tasks'):
        validate_tensordata_auto_composition(TaskTypesEnum.ts_forecasting, DataTypesEnum.table)

    with pytest.raises(ValueError, match='supports only tabular data'):
        validate_tensordata_auto_composition(TaskTypesEnum.classification, DataTypesEnum.ts)

