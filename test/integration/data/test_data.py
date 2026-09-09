import numpy as np

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.utils import fedot_project_root


def test_data_from_image():
    data_path = fedot_project_root().joinpath('test', 'data')
    dataset_to_validate = InputData.from_image(
        images=str(data_path.joinpath('test_data.npy')),
        labels=str(data_path.joinpath('test_labels.npy')),
    )

    assert dataset_to_validate.data_type == DataTypesEnum.image
    assert isinstance(dataset_to_validate.features, np.ndarray)
    assert isinstance(dataset_to_validate.target, np.ndarray)
