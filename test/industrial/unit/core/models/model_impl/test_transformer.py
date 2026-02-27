import numpy as np
import pytest
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.operation.dummy.dummy_operation import init_input_data

from fedot_ind.core.models.nn.network_impl.common_model.transformer import TransformerModel


@pytest.fixture(scope='session')
def input_data():
    np.random.seed(34)
    features = np.random.rand(10, 4, 3)
    target = np.random.randint(0, 2, 10)
    return init_input_data(features, target)


def test_transformer_model(input_data):
    model = TransformerModel(params=OperationParameters(num_classes=2,
                                                        epochs=5,
                                                        batch_size=32)
                             )

    model.fit(input_data)
    pred = model._predict_model(input_data.features)

    assert model is not None
    assert pred.predict.shape[0] == input_data.features.shape[0]
