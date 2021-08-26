from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from fedot.core.operations.model import Model
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from data.data_manager import data_setup


def model_metrics_info(class_name, y_true, y_pred):
    print('\n', f'#test_eval_strategy_{class_name}')
    print(classification_report(y_true, y_pred))
    print('Test model accuracy: ', accuracy_score(y_true, y_pred))


def test_node_factory_log_reg_correct():
    model_type = 'logit'
    node = PrimaryNode(operation_type=model_type)

    expected_model = Model(operation_type=model_type).__class__
    actual_model = node.operation.__class__

    assert node.__class__ == PrimaryNode
    assert expected_model == actual_model


def test_eval_strategy_logreg():
    train, test = data_setup()
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


def test_ordered_subnodes_hierarchy():
    first_node = PrimaryNode('knn')
    second_node = PrimaryNode('knn')
    third_node = SecondaryNode('lda', nodes_from=[first_node, second_node])
    root = SecondaryNode('logit', nodes_from=[third_node])

    ordered_nodes = root.ordered_subnodes_hierarchy()

    assert len(ordered_nodes) == 4


def test_distance_to_primary_level():
    first_node = PrimaryNode('knn')
    second_node = PrimaryNode('knn')
    third_node = SecondaryNode('lda', nodes_from=[first_node, second_node])
    root = SecondaryNode('logit', nodes_from=[third_node])

    distance = root.distance_to_primary_level

    assert distance == 2
