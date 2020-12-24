import numpy as np
import pytest
from sklearn.datasets import load_iris

from fedot.core.chains.node import PrimaryNode
from fedot.core.data.data import InputData
from fedot.core.data.preprocessing import Normalization, TextPreprocessingStrategy
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
