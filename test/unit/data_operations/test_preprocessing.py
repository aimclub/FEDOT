import math
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from sklearn.datasets import load_iris, make_regression
from sklearn.preprocessing import StandardScaler

from fedot.core.chains.node import PrimaryNode
from fedot.core.data.data import InputData
from fedot.core.data.preprocessing import Normalization, TextPreprocessingStrategy, ScalingWithImputation
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture()
def data_setup() -> InputData:
    predictors, response = load_iris(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)
    return data


@pytest.fixture()
def np_array_regression_with_missing_values() -> np.array:
    np.random.seed(42)
    features, target = make_regression(n_samples=1024, n_features=20, shuffle=True)

    data = pd.DataFrame(features)
    mask = np.random.choice([True, False], p=[0.1, 0.9], size=data.shape)
    data = data.mask(mask).to_numpy()

    return data


def test_scaling_with_imputation(np_array_regression_with_missing_values):
    scaler = ScalingWithImputation()
    actual_scaled_data = scaler.fit_apply(np_array_regression_with_missing_values)

    scaler = StandardScaler()
    df = pd.DataFrame(np_array_regression_with_missing_values)
    data = df.fillna(df.mean()).to_numpy()
    expected_scaled_data = scaler.fit_transform(data)

    result = []
    for i in range(len(actual_scaled_data)):
        for j in range(len(actual_scaled_data[0])):
            result.append(math.isclose(actual_scaled_data[i][j], expected_scaled_data[i][j], abs_tol=0.00001))

    assert all(result)


def test_node_with_manual_preprocessing_has_correct_behaviour_and_attributes(data_setup):
    model_type = 'logit'

    node_default = PrimaryNode(model_type=model_type)
    node_manual = PrimaryNode(model_type=model_type, manual_preprocessing_func=Normalization)

    default_node_prediction = node_default.fit(data_setup)
    manual_node_prediction = node_manual.fit(data_setup)

    assert node_manual.manual_preprocessing_func is not None
    assert node_default.manual_preprocessing_func != node_manual.manual_preprocessing_func

    assert not np.array_equal(default_node_prediction.predict, manual_node_prediction.predict)

    assert node_manual.descriptive_id == '/n_logit_default_params_custom_preprocessing=Normalization'


def test_image_preprocessing_strategy():

    training_set, testing_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    task = Task(TaskTypesEnum.classification)

    if type(training_set) is tuple:
        training_image, test_image = training_set[0][:5], training_set[1][:5]

    dataset_to_train = InputData.from_image(images=training_image, labels=test_image, task=task, aug_flag=False)

    assert dataset_to_train.task.task_type == TaskTypesEnum.classification
    assert dataset_to_train.data_type.name == 'image'

def test_text_preprocessing_strategy():
    test_text = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]

    preproc_strategy = TextPreprocessingStrategy()

    fit_result = preproc_strategy.fit(test_text)

    apply_result = preproc_strategy.apply(test_text)

    assert isinstance(fit_result, TextPreprocessingStrategy)
    assert apply_result[0] != test_text[0]