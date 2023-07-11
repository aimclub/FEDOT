import datetime
import os
import platform
import time
from copy import deepcopy
from multiprocessing import set_start_method

import numpy as np
import pytest
from sklearn.datasets import load_iris

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import probs_to_labels
from fedot.preprocessing.preprocessing import DataPreprocessor
from test.integration.composer.test_composer import to_categorical_codes
from test.integration.models.test_model import classification_dataset_with_redundant_features
from test.unit.dag.test_graph_operator import get_pipeline
from test.unit.pipelines.test_pipeline_comparison import pipeline_first
from test.unit.tasks.test_forecasting import get_ts_data


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
    input_data.idx = to_categorical_codes(categorical_ids=input_data.idx)
    return input_data


@pytest.mark.parametrize('data_fixture', ['data_setup', 'file_data_setup'])
def test_nodes_sequence_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train, _ = train_test_data_setup(data)

    first = PipelineNode(operation_type='logit')
    second = PipelineNode(operation_type='lda', nodes_from=[first])
    third = PipelineNode(operation_type='qda', nodes_from=[first])
    final = PipelineNode(operation_type='knn', nodes_from=[second, third])

    train_predicted = final.fit(input_data=train)

    assert final.descriptive_id == (
        '((/n_logit;)/'
        'n_lda;;(/'
        'n_logit;)/'
        'n_qda;)/'
        'n_knn')

    assert train_predicted.predict.shape[0] == train.target.shape[0]
    assert final.fitted_operation is not None


def test_pipeline_hierarchy_fit_correct(data_setup):
    data = data_setup
    train, _ = train_test_data_setup(data)

    first = PipelineNode(operation_type='logit')
    second = PipelineNode(operation_type='logit', nodes_from=[first])
    third = PipelineNode(operation_type='logit', nodes_from=[first])
    final = PipelineNode(operation_type='logit', nodes_from=[second, third])

    pipeline = Pipeline()
    for node in [first, second, third, final]:
        pipeline.add_node(node)

    pipeline.unfit()
    train_predicted = pipeline.fit(input_data=train)

    assert pipeline.root_node.descriptive_id == (
        '((/n_logit;)/'
        'n_logit;;(/'
        'n_logit;)/'
        'n_logit;)/'
        'n_logit')

    assert pipeline.length == 4
    assert pipeline.depth == 3
    assert train_predicted.predict.shape[0] == train.target.shape[0]
    assert final.fitted_operation is not None


def test_pipeline_sequential_fit_correct(data_setup):
    data = data_setup
    train, _ = train_test_data_setup(data)

    first = PipelineNode(operation_type='logit')
    second = PipelineNode(operation_type='logit', nodes_from=[first])
    third = PipelineNode(operation_type='logit', nodes_from=[second])
    final = PipelineNode(operation_type='logit', nodes_from=[third])

    pipeline = Pipeline()
    for node in [first, second, third, final]:
        pipeline.add_node(node)

    train_predicted = pipeline.fit(input_data=train)

    assert pipeline.root_node.descriptive_id == (
        '(((/n_logit;)/'
        'n_logit;)/'
        'n_logit;)/'
        'n_logit')

    assert pipeline.length == 4
    assert pipeline.depth == 4
    assert train_predicted.predict.shape[0] == train.target.shape[0]
    assert final.fitted_operation is not None


def test_pipeline_with_datamodel_fit_correct(data_setup):
    data = data_setup
    train_data, test_data = train_test_data_setup(data)

    pipeline = Pipeline()

    node_data = PipelineNode('logit')
    node_first = PipelineNode('bernb')
    node_second = PipelineNode('rf')

    node_second.nodes_from = [node_first, node_data]

    pipeline.add_node(node_data)
    pipeline.add_node(node_first)
    pipeline.add_node(node_second)

    pipeline.fit(train_data)
    results = np.asarray(probs_to_labels(pipeline.predict(test_data).predict))

    # Target for current case must be column
    test_data.target = test_data.target.reshape((-1, 1))
    assert results.shape == test_data.target.shape


def test_secondary_nodes_is_invariant_to_inputs_order(data_setup):
    data = data_setup
    # Preprocess data - determine features columns
    data = DataPreprocessor().obligatory_prepare_for_fit(data)
    train, test = train_test_data_setup(data)

    first = PipelineNode(operation_type='logit')
    second = PipelineNode(operation_type='lda')
    third = PipelineNode(operation_type='knn')
    final = PipelineNode(operation_type='logit',
                         nodes_from=[first, second, third])

    pipeline = Pipeline()
    for node in [first, second, third, final]:
        pipeline.add_node(node)

    first = deepcopy(first)
    second = deepcopy(second)
    third = deepcopy(third)

    final_shuffled = PipelineNode(operation_type='logit',
                                  nodes_from=[third, first, second])

    pipeline_shuffled = Pipeline()
    # change order of nodes in list
    for node in [final_shuffled, third, first, second]:
        pipeline_shuffled.add_node(node)

    train_predicted = pipeline.fit(input_data=train)

    train_predicted_shuffled = pipeline_shuffled.fit(input_data=train)

    # train results should be invariant
    assert pipeline.root_node.descriptive_id == pipeline_shuffled.root_node.descriptive_id
    assert np.equal(train_predicted.predict, train_predicted_shuffled.predict).all()

    test_predicted = pipeline.predict(input_data=test)
    test_predicted_shuffled = pipeline_shuffled.predict(input_data=test)

    # predict results should be invariant
    assert np.equal(test_predicted.predict, test_predicted_shuffled.predict).all()

    # change parents order for the nodes fitted pipeline
    nodes_for_change = pipeline.nodes[3].nodes_from
    pipeline.nodes[3].nodes_from = [nodes_for_change[2], nodes_for_change[0], nodes_for_change[1]]
    pipeline.nodes[3].unfit()
    pipeline.fit(train)
    test_predicted_re_shuffled = pipeline.predict(input_data=test)

    # predict results should be invariant
    assert np.equal(test_predicted.predict, test_predicted_re_shuffled.predict).all()


def test_pipeline_with_custom_params_for_model(data_setup):
    data = data_setup
    custom_params = dict(n_neighbors=1,
                         weights='uniform',
                         p=1)

    first = PipelineNode(operation_type='logit')
    second = PipelineNode(operation_type='lda')
    final = PipelineNode(operation_type='knn', nodes_from=[first, second])

    pipeline = Pipeline(final)
    pipeline_default_params = deepcopy(pipeline)

    pipeline.root_node.parameters = custom_params

    pipeline_default_params.fit(data)
    pipeline.fit(data)

    custom_params_prediction = pipeline.predict(data).predict
    default_params_prediction = pipeline_default_params.predict(data).predict

    assert not np.array_equal(custom_params_prediction, default_params_prediction)


def test_pipeline_with_wrong_data():
    pipeline = Pipeline(PipelineNode('linear'))
    data_seq = np.arange(0, 10)
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=10))

    data = InputData(idx=data_seq, features=data_seq, target=data_seq,
                     data_type=DataTypesEnum.ts, task=task)

    with pytest.raises(ValueError):
        pipeline.fit(data)


def test_pipeline_str():
    # given
    first = PipelineNode(operation_type='logit')
    second = PipelineNode(operation_type='lda')
    third = PipelineNode(operation_type='knn')
    final = PipelineNode(operation_type='rf',
                         nodes_from=[first, second, third])
    pipeline = Pipeline()
    pipeline.add_node(final)

    expected_pipeline_description = "{'depth': 2, 'length': 4, 'nodes': [rf, logit, lda, knn]}"

    # when
    actual_pipeline_description = str(pipeline)

    # then
    assert actual_pipeline_description == expected_pipeline_description


def test_pipeline_repr():
    first = PipelineNode(operation_type='logit')
    second = PipelineNode(operation_type='lda')
    third = PipelineNode(operation_type='knn')
    final = PipelineNode(operation_type='rf',
                         nodes_from=[first, second, third])
    pipeline = Pipeline()
    pipeline.add_node(final)

    expected_pipeline_description = "{'depth': 2, 'length': 4, 'nodes': [rf, logit, lda, knn]}"

    assert repr(pipeline) == expected_pipeline_description


def test_update_node_in_pipeline_correct():
    first = PipelineNode(operation_type='logit')
    final = PipelineNode(operation_type='rf', nodes_from=[first])

    pipeline = Pipeline()
    pipeline.add_node(final)
    new_node = PipelineNode('svc')
    replacing_node = PipelineNode('logit', nodes_from=[new_node])

    pipeline.update_node(old_node=first, new_node=replacing_node)

    assert replacing_node in pipeline.nodes
    assert new_node in pipeline.nodes
    assert first not in pipeline.nodes


def test_delete_node_with_redirection():
    first = PipelineNode(operation_type='logit')
    second = PipelineNode(operation_type='lda')
    third = PipelineNode(operation_type='knn', nodes_from=[first, second])
    final = PipelineNode(operation_type='rf',
                         nodes_from=[third])
    pipeline = Pipeline()
    pipeline.add_node(final)

    pipeline.delete_node(third)

    assert pipeline.length == 3
    assert first in pipeline.root_node.nodes_from


def test_delete_primary_node():
    # given
    first = PipelineNode(operation_type='logit')
    second = PipelineNode(operation_type='lda')
    third = PipelineNode(operation_type='knn', nodes_from=[first])
    final = PipelineNode(operation_type='rf',
                         nodes_from=[second, third])
    pipeline = Pipeline(final)

    # when
    pipeline.delete_node(first)

    new_primary_node = [node for node in pipeline.nodes if node.operation.operation_type == 'knn'][0]

    # then
    assert pipeline.length == 3
    assert isinstance(new_primary_node, PipelineNode)
    assert new_primary_node.is_primary


def test_update_subtree():
    # given
    pipeline = get_pipeline()
    subroot_parent = PipelineNode('rf')
    subroot = PipelineNode('rf', nodes_from=[subroot_parent])
    node_to_replace = pipeline.nodes[2]

    # when
    pipeline.update_subtree(node_to_replace, subroot)

    # then
    assert pipeline.nodes[2].operation.operation_type == 'rf'
    assert pipeline.nodes[3].operation.operation_type == 'rf'


def test_delete_subtree():
    # given
    pipeline = get_pipeline()
    subroot = pipeline.nodes[2]

    # when
    pipeline.delete_subtree(subroot)

    # then
    assert pipeline.length == 3


def test_pipeline_fit_time_constraint():
    system = platform.system()
    if system == 'Linux':
        set_start_method("spawn", force=True)
    data = classification_dataset_with_redundant_features()
    train_data, test_data = train_test_data_setup(data=data)
    test_pipeline_first = pipeline_first()

    time_constraint = datetime.timedelta(seconds=0)
    predicted_first = None
    computation_time_first = None
    process_start_time = time.time()
    try:
        predicted_first = test_pipeline_first.fit(input_data=train_data, time_constraint=time_constraint)
    except Exception as ex:
        received_ex = ex
        computation_time_first = test_pipeline_first.computation_time
        assert type(received_ex) is TimeoutError
    comp_time_proc_with_first_constraint = (time.time() - process_start_time)

    time_constraint = datetime.timedelta(seconds=3)
    process_start_time = time.time()
    try:
        test_pipeline_first.fit(input_data=train_data, time_constraint=time_constraint)
    except Exception as ex:
        received_ex = ex
        assert type(received_ex) is TimeoutError
    comp_time_proc_with_second_constraint = (time.time() - process_start_time)

    test_pipeline_second = pipeline_first()
    predicted_second = test_pipeline_second.fit(input_data=train_data,
                                                time_constraint=datetime.timedelta(seconds=2.1))
    computation_time_second = test_pipeline_second.computation_time
    assert comp_time_proc_with_first_constraint < comp_time_proc_with_second_constraint
    assert computation_time_first is None
    assert predicted_first is None
    assert computation_time_second is not None
    assert predicted_second is not None


@pytest.mark.parametrize('data_fixture', ['data_setup', 'file_data_setup'])
def test_pipeline_unfit(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    pipeline = Pipeline(PipelineNode('logit'))
    pipeline.fit(data)
    assert pipeline.is_fitted

    pipeline.unfit()
    assert not pipeline.is_fitted
    assert not pipeline.root_node.fitted_operation

    with pytest.raises(ValueError):
        assert pipeline.predict(data)


def test_ts_forecasting_pipeline_with_poly_features():
    """ Test pipeline with polynomial features in ts forecasting task """
    lagged_node = PipelineNode('lagged')
    poly_node = PipelineNode('poly_features', nodes_from=[lagged_node])
    ridge_node = PipelineNode('ridge', nodes_from=[poly_node])
    pipeline = Pipeline(ridge_node)

    train_data, test_data = get_ts_data(n_steps=25, forecast_length=5)

    pipeline.fit(train_data)
    prediction = pipeline.predict(test_data)
    assert prediction is not None


def test_get_nodes_with_operation():
    pipeline = pipeline_first()
    actual_nodes = pipeline.get_nodes_by_name(name='rf')
    expected_nodes = [pipeline.nodes[2], pipeline.nodes[-1]]

    assert (actual is expected for actual, expected in zip(actual_nodes, expected_nodes))


def test_get_node_with_uid():
    pipeline = pipeline_first()

    uid_of_first_node = pipeline.nodes[0].uid
    actual_node = pipeline.get_node_by_uid(uid=uid_of_first_node)
    expected_node = pipeline.nodes[0]
    assert actual_node is expected_node

    uid_of_non_existent_node = '123456789'
    assert pipeline.get_node_by_uid(uid=uid_of_non_existent_node) is None
