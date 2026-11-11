import numpy as np
import pytest
import torch

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.multimodal.supplementary_data import SupplementaryData
from fedot.core.optimisers.objective.oof_objective_rules import build_oof_metric_data
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.mark.unit
def test_build_oof_metric_data_adapts_legacy_data_to_tensor_metric_contract():
    task = Task(TaskTypesEnum.classification)
    reference = InputData(
        idx=np.array([0, 1]),
        features=np.array([[1.0], [2.0]]),
        target=np.array([0, 1]),
        task=task,
        data_type=DataTypesEnum.table,
        supplementary_data=SupplementaryData(),
    )
    predicted = OutputData(
        idx=np.array([0, 1]),
        features=np.array([[1.0], [2.0]]),
        predict=np.array([[0.8], [0.9]]),
        target=np.array([0, 1]),
        task=task,
        data_type=DataTypesEnum.table,
        supplementary_data=SupplementaryData(),
    )

    metric_data = build_oof_metric_data(reference, predicted)

    assert torch.equal(metric_data.reference.target, torch.tensor([0, 1]))
    assert torch.equal(metric_data.predicted.predict, torch.tensor([[0.8], [0.9]], dtype=torch.float64))
    assert np.array_equal(metric_data.reference.idx, reference.idx)
