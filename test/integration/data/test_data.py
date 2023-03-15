import numpy as np

from fedot.core.repository.dataset_types import DataTypesEnum
from test.unit.tasks.test_classification import get_image_classification_data


def test_data_from_image():
    _, _, dataset_to_validate = get_image_classification_data()

    assert dataset_to_validate.data_type == DataTypesEnum.image
    assert type(dataset_to_validate.features) == np.ndarray
    assert type(dataset_to_validate.target) == np.ndarray
