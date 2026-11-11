from unittest.mock import Mock

import numpy as np
import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.data_operations.sklearn_filters import NonLinearRegRANSACImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def test_ransac_falls_back_to_unfiltered_data_when_inliers_ratio_is_too_low():
    data = TensorData(idx=torch.arange(5),
                     features=torch.arange(5.).reshape(-1, 1),
                     target=torch.arange(5.),
                     task=Task(TaskTypesEnum.regression),
                     data_type=DataTypesEnum.table)
    implementation = NonLinearRegRANSACImplementation(OperationParameters(residual_threshold=0.1))

    def fit_with_too_few_inliers(*args, **kwargs):
        implementation.operation.inlier_mask_ = np.array([True, False, False, False, False])

    implementation.operation.fit = Mock(side_effect=fit_with_too_few_inliers)

    implementation.fit(data)
    transformed_data = implementation.transform_for_fit(data)

    assert implementation.operation.fit.call_count == implementation.max_iter
    assert implementation.operation.inlier_mask_ is None
    torch.testing.assert_close(transformed_data.idx, data.idx)
    torch.testing.assert_close(transformed_data.features, data.features)
