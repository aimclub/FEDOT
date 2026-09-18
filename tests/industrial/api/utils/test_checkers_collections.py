import numpy as np

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.industrial.api.utils.checkers_collections import DataCheck


def test_data_check_uses_rule_based_data_type_and_detection_window_task():
    checker = DataCheck(
        input_data=(np.arange(12).reshape(6, 2), np.array([0, 1, 0, 1, 0, 1])),
        task='classification',
        industrial_task_params={
            'data_type': 'time_series', 'detection_window': 4},
    )

    result = checker._transformation_for_other_task({
        'features': np.arange(12).reshape(6, 2),
        'target': np.array([0, 1, 0, 1, 0, 1]).reshape(-1, 1),
        'multi_features': False,
        'multi_target': False,
    })

    assert checker.data_type == DataTypesEnum.ts
    assert checker.data_type_plan.tensor_canonical_data_type == DataTypesEnum.ts
    assert result.task.task_type is TaskTypesEnum.ts_forecasting
    assert result.task.task_params.forecast_length == 4
