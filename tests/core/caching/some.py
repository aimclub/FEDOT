from fedot.core.caching.saver import Saver
from fedot.core.data.tensor_data import TensorDataCreator
import numpy as np


def test_saver_saves_tensor_data_as_normalized_torch_payload():
    features = np.random.rand(100, 10)

    td = TensorDataCreator.create(
        features,
        backend_name="cpu"
    )
    Saver.save(td, "tensor_one")


if __name__ == "__main__":
    test_saver_saves_tensor_data_as_normalized_torch_payload()