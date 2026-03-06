import numpy as np
import pandas as pd

from fedot.api.sampling_stage.providers import SamplingZooProvider


def test_parse_partition_value_from_indices_dict():
    parsed = SamplingZooProvider._parse_partition_value({'indices': [1, 3, 5]})
    assert np.array_equal(parsed, np.array([1, 3, 5]))


def test_parse_partition_value_from_feature_dataframe_index():
    frame = pd.DataFrame({'a': [10, 20]}, index=[4, 7])
    parsed = SamplingZooProvider._parse_partition_value({'feature': frame})
    assert np.array_equal(parsed, np.array([4, 7]))


def test_parse_partition_value_from_list():
    parsed = SamplingZooProvider._parse_partition_value([0, 2, 6])
    assert np.array_equal(parsed, np.array([0, 2, 6]))
