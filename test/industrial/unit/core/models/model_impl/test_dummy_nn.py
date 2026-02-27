import torch

from fedot.industrial.core.models.nn.network_impl.common_model.dummy_nn import DummyOverComplicatedNeuralNetwork


def test_dummy_nn():
    dummy = DummyOverComplicatedNeuralNetwork(input_dim=1000, output_dim=10)
    image = torch.randn(1, 1000, 1000)
    output = dummy(image)
    assert output.shape == (1, 10)
