import numpy as np
from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
from torch import Tensor

from fedot_ind.core.models.nn.network_impl.feature_extraction.mini_rocket import MiniRocket, MiniRocketExtractor, \
    get_minirocket_features, \
    MiniRocketHead


def test_mini_rocket():
    mini_rocket = MiniRocket(input_dim=1000,
                             output_dim=10,
                             seq_len=1000,
                             num_features=10_000)
    input_tensor = Tensor(np.random.rand(1, 1000, 1000))
    mini_rocket.fit(input_tensor)


def test_get_minirocket_features():
    input_tensor = Tensor(np.random.rand(100, 100, 100))
    model = MiniRocket(input_dim=100,
                       output_dim=10,
                       seq_len=100,
                       num_features=100)
    features = get_minirocket_features(data=input_tensor,
                                       model=model,
                                       chunksize=10)
    assert features.shape == (100, 10, 1)


def test_mini_rocket_head():
    head = MiniRocketHead(input_dim=1000,
                          output_dim=10)

    assert head(input=Tensor(np.random.rand(2, 1000))).shape == (2, 10)


def test_mini_rocket_extractor():
    extractor = MiniRocketExtractor({'num_features': 100})
    input_features = np.random.rand(100, 3, 100)
    input_target = np.random.randint(0, 2, 100)
    input_data = init_input_data(input_features, input_target)
    features = extractor.transform(input_data)
    assert features.features.shape == (100, 3, 100)
