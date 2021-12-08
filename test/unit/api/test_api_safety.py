import numpy as np

from fedot.api.api_utils.api_safety import ApiSafety
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task


def get_safety_with_specific_params():
    safety_module = ApiSafety(safe_mode=True)
    safety_module.max_cat_cardinality = 5
    safety_module.max_size = 10
    return safety_module


def get_small_cat_data():
    features = np.array([
        ["a", "qq", 0.5],
        ["b", "pp", 1],
        ["c", "oo", 3],
        ["d", "oo", 3]
    ], dtype=object)
    target = np.array([0, 1, 1])
    return InputData(idx=features.shape,
                     features=features,
                     target=target,
                     data_type=DataTypesEnum.table,
                     task=Task(TaskTypesEnum.classification)
                     )


def test_safety_correct():
    api_safety = get_safety_with_specific_params()
    data = get_small_cat_data()
    data = api_safety.safe_preprocess(data)
    assert data.features.shape[0] * data.features.shape[1] < api_safety.max_size
    assert data.features.shape[1] == 3
    assert data.features[0, 0] == 0


def test_safety_do_not_needed_correct():
    api_safety = get_safety_with_specific_params()
    api_safety.max_cat_cardinality = 100
    api_safety.max_size = 100
    data = get_small_cat_data()
    data = api_safety.safe_preprocess(data)
    assert data.features.shape[0] * data.features.shape[1] == 12
    assert data.features.shape[1] == 3
    assert data.features[0, 0] == "a"
