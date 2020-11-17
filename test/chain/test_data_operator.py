import numpy as np
from sklearn.datasets import load_iris

from core.composer.chain import Chain
from core.composer.data_operator import DataOperator
from core.composer.node import PrimaryNode
from core.models.data import InputData, train_test_data_setup
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import TaskTypesEnum, Task


def data_setup():
    predictors, response = load_iris(return_X_y=True)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)
    return data


def blank_input_data():
    empty_data = InputData(idx=np.arange(0, 10),
                           features=np.zeros((10, 1)),
                           target=np.zeros((10, 1)),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)

    return empty_data


def test_single_node_with_operator_fit_correct():
    model_type = 'xgboost'
    data = data_setup()
    train, _ = train_test_data_setup(data)
    operator = DataOperator(external_source=train)
    node = PrimaryNode(model_type=model_type,
                       data_operator=operator)

    chain = Chain(nodes=[node])
    empty = blank_input_data()
    train_predicted = chain.fit(input_data=empty)
    train_another_predicted = chain.predict(input_data=empty)

    assert len(train_predicted.predict) == len(train.idx)
    assert len(train_another_predicted.predict) == len(train.idx)
