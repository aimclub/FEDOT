import os

import numpy as np
import pandas as pd
import pytest

from core.models.data import Data


@pytest.fixture()
def output_dataset():
    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    threshold = 0.5
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = Data(idx=np.arange(0, 100), features=x, target=classes)

    return data


def test_data_from_csv():
    test_file_path = str(os.path.dirname(__file__))
    file = 'data/test_dataset.csv'
    df = pd.read_csv(os.path.join(test_file_path, file))
    data_array = np.array(df).T
    features = data_array[1:-1].T
    target = data_array[-1]
    idx = data_array[0]
    expected_features = Data(features=features, target=target, idx=idx).features.all()
    actual_features = Data.from_csv(os.path.join(test_file_path, file)).features.all()
    assert expected_features == actual_features


def test_data_from_predictions(output_dataset):
    data_1 = output_dataset
    data_2 = output_dataset
    data_3 = output_dataset
    target = output_dataset.target
    new_input_data = Data.from_predictions(outputs=[data_1, data_2, data_3],
                                           target=target)
    assert new_input_data.features.all() == np.array(
        [data_1.target, data_2.target, data_3.target]).all()
