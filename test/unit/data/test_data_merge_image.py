import pytest
from typing import Tuple, Iterable
import numpy as np

from fedot.core.data.data import OutputData
from fedot.core.data.data_merger import DataMerger
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.array_utilities import atleast_4d


def generate_output_images(image_sizes: Iterable[Tuple[int, int]], num_samples=10):
    num_classes = 5
    task = Task(TaskTypesEnum.classification)
    data_type = DataTypesEnum.image
    target = np.random.randint(0, num_classes, (num_samples, 1))

    outputs = []
    for img_shape in image_sizes:
        idx = np.arange(0, num_samples)
        features = np.random.random_sample((num_samples, *img_shape))
        predict = 0.9 * features
        output_data = OutputData(idx=idx, features=features, predict=predict, target=target,
                                 task=task, data_type=data_type,)
        outputs.append(output_data)
    return outputs


test_image_sizes = (
    ((16, 16, 1),),
    ((16, 24,),),
    ((16, 16, 1), (16, 16,)),
    ((16, 16, 3), (16, 16, 2)),
    ((16, 24,), (16, 24, 2), (16, 24, 1)),
    ((8, 8, 1), (16, 16, 1), (24, 24, 1)),
    ((8, 8, 2), (16, 16, 3), (24, 24, 1)),
    ((16, 8,), (8, 24, 1)),
)


@pytest.fixture(params=test_image_sizes, ids=lambda sizes: f'images sizes: {sizes}')
def output_images(request):
    return generate_output_images(request.param)


def test_data_merge_images(output_images):
    def get_num_channels(output_image: OutputData):
        return atleast_4d(output_image.predict).shape[3]

    img_wh = [img.predict.shape[1:3] for img in output_images]
    invalid_sizes = len(set(img_wh)) > 1  # Can merge only images of the same size
    expected_channels = sum(map(get_num_channels, output_images))
    expected_shape = (*img_wh[0], expected_channels)

    if invalid_sizes:
        with pytest.raises(ValueError, match='different sizes'):
            merged_image = DataMerger.get(output_images).merge()
    else:
        merged_image = DataMerger.get(output_images).merge()
        assert merged_image.features.shape[1:] == expected_shape
