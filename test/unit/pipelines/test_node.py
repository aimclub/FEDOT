import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from fedot.core.dag.graph_node import ordered_subnodes_hierarchy, DAGNode
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.model import Model
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


GraphNode = DAGNode


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


def model_metrics_info(class_name, y_true, y_pred):
    print('\n', f'#test_eval_strategy_{class_name}')
    print(classification_report(y_true, y_pred))
    print('Test model accuracy: ', accuracy_score(y_true, y_pred))


def test_node_factory_log_reg_correct(data_setup):
    model_type = 'logit'
    node = PrimaryNode(operation_type=model_type)

    expected_model = Model(operation_type=model_type).__class__
    actual_model = node.operation.__class__

    assert node.__class__ == PrimaryNode
    assert expected_model == actual_model


def test_eval_strategy_logreg(data_setup):
    data_set = data_setup
    train, test = train_test_data_setup(data=data_set)
    test_skl_model = LogisticRegression(C=10., random_state=1,
                                        solver='liblinear',
                                        max_iter=10000, verbose=0)
    test_skl_model.fit(train.features, train.target)
    expected_result = test_skl_model.predict(test.features)

    test_model_node = PrimaryNode(operation_type='logit')
    test_model_node.fit(input_data=train)
    actual_result = test_model_node.predict(input_data=test)

    assert len(actual_result.predict) == len(expected_result)


def test_node_str():
    # given
    operation_type = 'logit'
    test_model_node = PrimaryNode(operation_type=operation_type)
    expected_node_description = operation_type

    # when
    actual_node_description = str(test_model_node)

    # then
    assert actual_node_description == expected_node_description


def test_node_repr():
    # given
    operation_type = 'logit'
    test_model_node = PrimaryNode(operation_type=operation_type)
    expected_node_description = operation_type

    # when
    actual_node_description = repr(test_model_node)

    # then
    assert actual_node_description == expected_node_description


def test_node_repr_with_params():
    # given
    operation_type = 'logit'
    params = {'some_param': 10}
    test_model_node = GraphNode(dict(name=operation_type, params=params))
    expected_node_description = f'n_{operation_type}_{params}'

    # when
    actual_node_description = test_model_node.description()

    # then
    assert actual_node_description == expected_node_description


def test_ordered_subnodes_hierarchy():
    first_node = PrimaryNode('knn')
    second_node = PrimaryNode('knn')
    third_node = SecondaryNode('lda', nodes_from=[first_node, second_node])
    root = SecondaryNode('logit', nodes_from=[third_node])

    ordered_nodes = ordered_subnodes_hierarchy(root)

    assert len(ordered_nodes) == 4
    assert ordered_nodes == [root, third_node, first_node, second_node]


def test_ordered_subnodes_cycle():
    cycle_node = GraphNode('knn')
    second_node = GraphNode('knn')
    third_node = GraphNode('lda', nodes_from=[cycle_node, second_node])
    root = GraphNode('logit', nodes_from=[third_node])
    cycle_node.nodes_from = [root]

    with pytest.raises(ValueError, match='cycle'):
        ordered_subnodes_hierarchy(root)


def test_distance_to_primary_level():
    first_node = PrimaryNode('knn')
    second_node = PrimaryNode('knn')
    third_node = SecondaryNode('lda', nodes_from=[first_node, second_node])
    root = SecondaryNode('logit', nodes_from=[third_node])

    distance = root.distance_to_primary_level

    assert distance == 2


def test_node_return_correct_operation_info():
    node = PrimaryNode('simple_imputation')
    operation_tags = node.tags

    correct_tags = ["simple", "imputation"]
    assert all(correct_tag in operation_tags for correct_tag in correct_tags)
