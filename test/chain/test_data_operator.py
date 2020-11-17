from copy import copy

import numpy as np
from sklearn.datasets import load_iris

from core.composer.chain import Chain
from core.composer.data_operator import DataOperator
from core.composer.node import PrimaryNode, SecondaryNode, CachedState
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


def fit_model_in_primary(node: PrimaryNode):
    input_data = node.operator.input()
    transformed = node._transform(input_data)
    preprocessed_data, preproc_strategy = node._preprocess(transformed)

    if not node.cache.actual_cached_state:
        print('Cache is not actual')
        cached_model, model_predict = node.model.fit(data=preprocessed_data)
        node.cache.append(CachedState(preprocessor=copy(preproc_strategy),
                                      model=cached_model))
    else:
        print('Model were obtained from cache')

        model_predict = node.model.predict(fitted_model=node.cache.actual_cached_state.model,
                                           data=preprocessed_data)

    final_output = node.output_from_prediction(input_data, model_predict)
    node.operator.set_output(output=final_output)

    return final_output


def test_fit_model_in_primary_correct():
    model_type = 'xgboost'
    data = data_setup()
    train, _ = train_test_data_setup(data)
    operator = DataOperator(external_source=train, type='external')
    node = PrimaryNode(model_type=model_type,
                       data_operator=operator)

    predicted = fit_model_in_primary(node)

    print(predicted)


def test_single_node_with_operator_fit_correct():
    model_type = 'xgboost'
    data = data_setup()
    train, _ = train_test_data_setup(data)
    operator = DataOperator(external_source=train, type='external')
    node = PrimaryNode(model_type=model_type,
                       data_operator=operator)
    operator.attached_node = node
    chain = Chain(nodes=[node])
    empty = blank_input_data()
    train_predicted = chain.fit(input_data=empty)
    train_another_predicted = chain.predict(input_data=empty)

    assert len(train_predicted.predict) == len(train.idx)
    assert len(train_another_predicted.predict) == len(train.idx)


def test_two_layers_chain_with_operator_fit_correct():
    data = data_setup()
    train, _ = train_test_data_setup(data)
    operator_primary = DataOperator(external_source=train, type='external')
    operator_secondary = DataOperator(external_source=None, type='from_parents')

    node_first = PrimaryNode(model_type='xgboost',
                             data_operator=operator_primary)
    operator_primary.attached_node = node_first
    node_second = SecondaryNode(model_type='logit', nodes_from=[node_first],
                                data_operator=operator_secondary)
    operator_secondary.attached_node = node_second

    chain = Chain()
    chain.add_node(node_second)

    train_predicted = chain.fit(input_data=train)
