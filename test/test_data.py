import os

import numpy as np
import pandas as pd
import pytest

from core.models.data import Data, OutputData, InputData
from core.repository.task_types import MachineLearningTasksEnum


@pytest.fixture()
def output_dataset():
    task_type = MachineLearningTasksEnum.classification
    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    threshold = 0.5
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = OutputData(idx=np.arange(0, 100), features=x, predict=classes,
                      task_type=task_type)

    return data


def test_data_from_csv():
    test_file_path = str(os.path.dirname(__file__))
    file = 'data/test_dataset.csv'
    task_type = MachineLearningTasksEnum.classification
    df = pd.read_csv(os.path.join(test_file_path, file))
    data_array = np.array(df).T
    features = data_array[1:-1].T
    target = data_array[-1]
    idx = data_array[0]
    expected_features = InputData(features=features, target=target,
                                  idx=idx,
                                  task_type=task_type).features.all()
    actual_features = InputData.from_csv(
        os.path.join(test_file_path, file)).features.all()
    assert expected_features == actual_features


def test_data_from_predictions(output_dataset):
    data_1 = output_dataset
    data_2 = output_dataset
    data_3 = output_dataset
    target = output_dataset.predict
    new_input_data = Data.from_predictions(outputs=[data_1, data_2, data_3],
                                           target=target)
    assert new_input_data.features.all() == np.array(
        [data_1.predict, data_2.predict, data_3.predict]).all()


def test_string_features_from_csv():
    test_file_path = str(os.path.dirname(__file__))
    file = 'data/scoring_train_cat.csv'
    expected_features = InputData.from_csv(os.path.join(test_file_path, file)).features

    assert expected_features.dtype == float
    assert np.isfinite(expected_features).all()
