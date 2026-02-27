import pytest

from fedot_ind.core.models.nn.network_impl.common_model.resnet import *


@pytest.mark.parametrize('model_name, model',
                         [(k, v) for k, v in CLF_MODELS.items()])
def test_resnet(model, model_name):
    model = model()
    assert model is not None
