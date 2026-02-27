import numpy as np
import pandas as pd
import pytest

from fedot_ind.core.models.nn.network_impl.common_model.inception import InceptionTimeModel
from fedot_ind.core.operation.dummy.dummy_operation import init_input_data


@pytest.fixture
def ts():
    features = pd.DataFrame(np.random.rand(100))
    target = np.random.rand(100)
    return init_input_data(X=features, y=target, task='regression')


def test_inception_time_model(ts):
    inception = InceptionTimeModel()
    loss_fn, optimizer = inception._init_model(ts=ts)
    assert loss_fn is not None
    assert optimizer is not None
