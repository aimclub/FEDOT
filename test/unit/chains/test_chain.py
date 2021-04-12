import datetime
import os
import platform
import time
from copy import deepcopy
from multiprocessing import set_start_method
from random import seed

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score as roc

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import probs_to_labels
from test.unit.chains.test_chain_comparison import chain_first
from test.unit.chains.test_chain_tuning import classification_dataset

seed(1)
np.random.seed(1)

tmp = classification_dataset


@pytest.fixture()
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


@pytest.fixture()
def classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../../data', 'advanced_classification.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.classification))


@pytest.fixture()
def file_data_setup():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/simple_classification.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    input_data.idx = _to_numerical(categorical_ids=input_data.idx)
    return input_data


def _to_numerical(categorical_ids: np.ndarray):
    encoded = pd.factorize(categorical_ids)[0]
    return encoded


@pytest.mark.parametrize('data_fixture', ['data_setup', 'file_data_setup'])
def test_nodes_sequence_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train, _ = train_test_data_setup(data)

    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='lda', nodes_from=[first])
    third = SecondaryNode(operation_type='qda', nodes_from=[first])
    final = SecondaryNode(operation_type='knn', nodes_from=[second, third])

    train_predicted = final.fit(input_data=train)

    assert final.descriptive_id == (
        '((/n_logit_default_params;)/'
        'n_lda_default_params;;(/'
        'n_logit_default_params;)/'
        'n_qda_default_params;)/'
        'n_knn_default_params')

    assert train_predicted.predict.shape[0] == train.target.shape[0]
    assert final.fitted_operation is not None


def test_chain_hierarchy_fit_correct(data_setup):
    data = data_setup
    train, _ = train_test_data_setup(data)

    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit', nodes_from=[first])
    third = SecondaryNode(operation_type='logit', nodes_from=[first])
    final = SecondaryNode(operation_type='logit', nodes_from=[second, third])

    chain = Chain()
    for node in [first, second, third, final]:
        chain.add_node(node)

    chain.unfit()
    train_predicted = chain.fit(input_data=train)

    assert chain.root_node.descriptive_id == (
        '((/n_logit_default_params;)/'
        'n_logit_default_params;;(/'
        'n_logit_default_params;)/'
        'n_logit_default_params;)/'
        'n_logit_default_params')

    assert chain.length == 4
    assert chain.depth == 3
    assert train_predicted.predict.shape[0] == train.target.shape[0]
    assert final.fitted_operation is not None


def test_chain_sequential_fit_correct(data_setup):
    data = data_setup
    train, _ = train_test_data_setup(data)

    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit', nodes_from=[first])
    third = SecondaryNode(operation_type='logit', nodes_from=[second])
    final = SecondaryNode(operation_type='logit', nodes_from=[third])

    chain = Chain()
    for node in [first, second, third, final]:
        chain.add_node(node)

    train_predicted = chain.fit(input_data=train, use_cache=False)

    assert chain.root_node.descriptive_id == (
        '(((/n_logit_default_params;)/'
        'n_logit_default_params;)/'
        'n_logit_default_params;)/'
        'n_logit_default_params')

    assert chain.length == 4
    assert chain.depth == 4
    assert train_predicted.predict.shape[0] == train.target.shape[0]
    assert final.fitted_operation is not None


def test_chain_with_datamodel_fit_correct(data_setup):
    data = data_setup
    train_data, test_data = train_test_data_setup(data)

    chain = Chain()

    node_data = PrimaryNode('logit')
    node_first = PrimaryNode('bernb')
    node_second = SecondaryNode('rf')

    node_second.nodes_from = [node_first, node_data]

    chain.add_node(node_data)
    chain.add_node(node_first)
    chain.add_node(node_second)

    chain.fit(train_data)
    results = np.asarray(probs_to_labels(chain.predict(test_data).predict))

    assert results.shape == test_data.target.shape


def test_secondary_nodes_is_invariant_to_inputs_order(data_setup):
    data = data_setup
    train, test = train_test_data_setup(data)

    first = PrimaryNode(operation_type='logit')
    second = PrimaryNode(operation_type='lda')
    third = PrimaryNode(operation_type='knn')
    final = SecondaryNode(operation_type='xgboost',
                          nodes_from=[first, second, third])

    chain = Chain()
    for node in [first, second, third, final]:
        chain.add_node(node)

    first = deepcopy(first)
    second = deepcopy(second)
    third = deepcopy(third)

    final_shuffled = SecondaryNode(operation_type='xgboost',
                                   nodes_from=[third, first, second])

    chain_shuffled = Chain()
    # change order of nodes in list
    for node in [final_shuffled, third, first, second]:
        chain_shuffled.add_node(node)

    train_predicted = chain.fit(input_data=train)

    train_predicted_shuffled = chain_shuffled.fit(input_data=train)

    # train results should be invariant
    assert chain.root_node.descriptive_id == chain_shuffled.root_node.descriptive_id
    assert np.equal(train_predicted.predict, train_predicted_shuffled.predict).all()

    test_predicted = chain.predict(input_data=test)
    test_predicted_shuffled = chain_shuffled.predict(input_data=test)

    # predict results should be invariant
    assert np.equal(test_predicted.predict, test_predicted_shuffled.predict).all()

    # change parents order for the nodes fitted chain
    nodes_for_change = chain.nodes[3].nodes_from
    chain.nodes[3].nodes_from = [nodes_for_change[2], nodes_for_change[0], nodes_for_change[1]]
    chain.nodes[3].unfit()
    chain.fit(train)
    test_predicted_re_shuffled = chain.predict(input_data=test)

    # predict results should be invariant
    assert np.equal(test_predicted.predict, test_predicted_re_shuffled.predict).all()


def test_chain_with_custom_params_for_model(data_setup):
    data = data_setup
    custom_params = dict(n_neighbors=1,
                         weights='uniform',
                         p=1)

    first = PrimaryNode(operation_type='logit')
    second = PrimaryNode(operation_type='lda')
    final = SecondaryNode(operation_type='knn', nodes_from=[first, second])

    chain = Chain()
    chain.add_node(final)
    chain_default_params = deepcopy(chain)

    chain.root_node.custom_params = custom_params

    chain_default_params.fit(data)
    chain.fit(data)

    custom_params_prediction = chain.predict(data).predict
    default_params_prediction = chain_default_params.predict(data).predict

    assert not np.array_equal(custom_params_prediction, default_params_prediction)


def test_chain_with_wrong_data():
    chain = Chain(PrimaryNode('linear'))
    data_seq = np.arange(0, 10)
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=10))

    data = InputData(idx=data_seq, features=data_seq, target=data_seq,
                     data_type=DataTypesEnum.ts, task=task)

    with pytest.raises(ValueError):
        chain.fit(data)


def test_chain_str():
    # given
    first = PrimaryNode(operation_type='logit')
    second = PrimaryNode(operation_type='lda')
    third = PrimaryNode(operation_type='knn')
    final = SecondaryNode(operation_type='xgboost',
                          nodes_from=[first, second, third])
    chain = Chain()
    chain.add_node(final)

    expected_chain_description = "{'depth': 2, 'length': 4, 'nodes': [xgboost, logit, lda, knn]}"

    # when
    actual_chain_description = str(chain)

    # then
    assert actual_chain_description == expected_chain_description


def test_chain_repr():
    first = PrimaryNode(operation_type='logit')
    second = PrimaryNode(operation_type='lda')
    third = PrimaryNode(operation_type='knn')
    final = SecondaryNode(operation_type='xgboost',
                          nodes_from=[first, second, third])
    chain = Chain()
    chain.add_node(final)

    expected_chain_description = "{'depth': 2, 'length': 4, 'nodes': [xgboost, logit, lda, knn]}"

    assert repr(chain) == expected_chain_description


def test_update_node_in_chain_raise_exception():
    first = PrimaryNode(operation_type='logit')
    final = SecondaryNode(operation_type='xgboost', nodes_from=[first])

    chain = Chain()
    chain.add_node(final)
    replacing_node = SecondaryNode('logit')

    with pytest.raises(ValueError) as exc:
        chain.update_node(old_node=first, new_node=replacing_node)

    assert str(exc.value) == "Can't update PrimaryNode with SecondaryNode"


def test_delete_node_with_redirection():
    first = PrimaryNode(operation_type='logit')
    second = PrimaryNode(operation_type='lda')
    third = SecondaryNode(operation_type='knn', nodes_from=[first, second])
    final = SecondaryNode(operation_type='xgboost',
                          nodes_from=[third])
    chain = Chain()
    chain.add_node(final)

    chain.delete_node(third)

    assert len(chain.nodes) == 3
    assert first in chain.root_node.nodes_from


def test_delete_primary_node_with_redirection():
    # given
    first = PrimaryNode(operation_type='logit')
    second = PrimaryNode(operation_type='lda')
    third = SecondaryNode(operation_type='knn', nodes_from=[first])
    final = SecondaryNode(operation_type='xgboost',
                          nodes_from=[second, third])
    chain = Chain()
    chain.add_node(final)

    # when
    chain.delete_node(first)

    new_primary_node = [node for node in chain.nodes if node.operation.operation_type == 'knn'][0]

    # then
    assert len(chain.nodes) == 3
    assert isinstance(new_primary_node, PrimaryNode)


def test_delete_secondary_node_with_multiple_children_and_redirection():
    # given
    logit_first = PrimaryNode(operation_type='logit')
    lda_first = PrimaryNode(operation_type='lda')
    knn_center = SecondaryNode(operation_type='knn', nodes_from=[logit_first, lda_first])
    logit_second = SecondaryNode(operation_type='logit', nodes_from=[knn_center])
    lda_second = SecondaryNode(operation_type='lda', nodes_from=[knn_center])
    final = SecondaryNode(operation_type='xgboost',
                          nodes_from=[logit_second, lda_second])

    chain = Chain()
    chain.add_node(final)

    # when
    chain.delete_node(knn_center)

    # then
    updated_logit_second_parents = chain.nodes[1].nodes_from
    updated_lda_second_parents = chain.nodes[4].nodes_from

    assert len(chain.nodes) == 5
    assert updated_logit_second_parents[0] is logit_first
    assert updated_logit_second_parents[1] is lda_first
    assert updated_lda_second_parents[0] is logit_first
    assert updated_lda_second_parents[1] is lda_first


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_chain_fit_time_constraint(data_fixture, request):
    system = platform.system()
    if system == 'Linux':
        set_start_method("spawn", force=True)
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)
    test_chain_first = chain_first()
    time_constraint = datetime.timedelta(minutes=0.01)
    predicted_first = None
    computation_time_first = None
    process_start_time = time.time()
    try:
        predicted_first = test_chain_first.fit(input_data=train_data, time_constraint=time_constraint)
    except Exception as ex:
        received_ex = ex
        computation_time_first = test_chain_first.computation_time
        assert type(received_ex) is TimeoutError
    comp_time_proc_with_first_constraint = (time.time() - process_start_time)
    time_constraint = datetime.timedelta(minutes=0.05)
    process_start_time = time.time()
    try:
        test_chain_first.fit(input_data=train_data, time_constraint=time_constraint)
    except Exception as ex:
        received_ex = ex
        assert type(received_ex) is TimeoutError
    comp_time_proc_with_second_constraint = (time.time() - process_start_time)
    test_chain_second = chain_first()
    predicted_second = test_chain_second.fit(input_data=train_data)
    computation_time_second = test_chain_second.computation_time
    assert comp_time_proc_with_first_constraint < comp_time_proc_with_second_constraint
    assert computation_time_first is None
    assert predicted_first is None
    assert computation_time_second is not None
    assert predicted_second is not None


def test_chain_fine_tune_all_nodes_correct(classification_dataset):
    data = classification_dataset

    first = PrimaryNode(operation_type='scaling')
    second = PrimaryNode(operation_type='knn')
    final = SecondaryNode(operation_type='dt', nodes_from=[first, second])

    chain = Chain(final)

    iterations_total, time_limit_minutes = 5, 1
    tuned_chain = chain.fine_tune_all_nodes(loss_function=roc, input_data=data,
                                            iterations=iterations_total,
                                            max_lead_time=time_limit_minutes)
    tuned_chain.predict(input_data=data)

    is_tuning_finished = True

    assert is_tuning_finished
