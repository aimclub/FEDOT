import torch

from fedot.core.operations.evaluation.operation_implementations.models.torch import TorchLinearClassifier
from fedot.core.operations.operation_parameters import OperationParameters


def test_torch_linear_reads_fit_params_from_operation_parameters():
    params = OperationParameters.from_operation_type("torch_linear", epochs=3)
    model = TorchLinearClassifier(params)
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])

    model.fit(features, target)

    assert params.get("epochs") == 3
    assert params.get("learning_rate") == 0.05
    assert model.module is not None
