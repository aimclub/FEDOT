import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData, split_train_test
from core.repository.model_types_repository import ModelTypesIdsEnum


@pytest.fixture()
def data_setup():
    predictors, response = load_breast_cancer(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    response = response[:100]
    predictors = predictors[:100]
    train_data_x, test_data_x = split_train_test(predictors)
    train_data_y, test_data_y = split_train_test(response)
    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=np.arange(0, len(train_data_y)))
    test_data = InputData(features=test_data_x, target=test_data_y,
                          idx=np.arange(0, len(test_data_y)))
    return train_data, test_data


def chain_first():
    #    XG
    #  |     \
    # XG      KNN
    # |  \    |  \
    # LR LDA LR  LDA
    chain = Chain()

    root_of_tree, root_child_first, root_child_second = \
        [NodeGenerator.secondary_node(model) for model in (ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.xgboost,
                                                           ModelTypesIdsEnum.knn)]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in (ModelTypesIdsEnum.logit, ModelTypesIdsEnum.lda):
            new_node = NodeGenerator.primary_node(requirement_model)
            root_node_child.nodes_from.append(new_node)
            chain.add_node(new_node)
        chain.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    chain.add_node(root_of_tree)
    return chain


def chain_second():
    chain = chain_first()
    new_node = NodeGenerator.secondary_node(ModelTypesIdsEnum.dt)
    for model_type in (ModelTypesIdsEnum.knn, ModelTypesIdsEnum.knn):
        new_node.nodes_from.append(NodeGenerator.primary_node(model_type))
    chain.replace_node(chain.root_node.nodes_from[0], new_node)
    return chain


def test_cache_model_changed(data_setup):
    """Changing the model in one of the tree node"""
    chain = chain_first()
    train, _ = data_setup
    chain.fit(input_data=train)
    function = ModelTypesIdsEnum.dt
    new_node = NodeGenerator.secondary_node(model_type=function)
    chain.update_node(old_node=chain.root_node.nodes_from[0], new_node=new_node)
    root = chain.root_node
    root_parent_first = root.nodes_from[0]
    root_parent_second = root.nodes_from[1]
    primary_node_first = root_parent_first.nodes_from[0]
    primary_node_second = root_parent_first.nodes_from[1]
    primary_node_third = root_parent_second.nodes_from[0]
    primary_node_fourth = root_parent_second.nodes_from[1]

    assert not all([root._is_cache_actual(), root_parent_first._is_cache_actual()])
    assert all([root_parent_second._is_cache_actual(), primary_node_first._is_cache_actual(),
                primary_node_second._is_cache_actual(), primary_node_third._is_cache_actual(),
                primary_node_fourth._is_cache_actual()])


def test_cache_subtree_changed(data_setup):
    """The subtree in source tree changed to other previously trained subtree"""
    train, _ = data_setup
    chain = chain_first()
    other_chain = chain_second()
    chain.fit(input_data=train)
    other_chain.fit(input_data=train)
    chain.replace_node(chain.root_node.nodes_from[0], other_chain.root_node.nodes_from[0])
    root = chain.root_node
    root_parent_first = root.nodes_from[0]
    root_parent_second = root.nodes_from[1]
    assert not root._is_cache_actual()
    assert all([root_parent_first._is_cache_actual(), root_parent_second._is_cache_actual()])


def test_cache_primary_node_changed_to_subtree(data_setup):
    """The primary node in source tree changed to other previously trained subtree"""
    train, _ = data_setup
    chain = chain_first()
    other_chain = chain_second()
    chain.fit(input_data=train)
    other_chain.fit(input_data=train)
    chain.replace_node(chain.root_node.nodes_from[0].nodes_from[0], other_chain.root_node.nodes_from[0])
    root = chain.root_node
    root_parent_first = root.nodes_from[0]
    root_parent_second = root.nodes_from[1]
    root_parent_first_parent_first = root_parent_first.nodes_from[0]
    root_parent_first_parent_second = root_parent_first.nodes_from[1]
    primary_node_first = root_parent_first_parent_first.nodes_from[0]
    primary_node_second = root_parent_first_parent_first.nodes_from[0]

    assert not all([root._is_cache_actual(), root_parent_first._is_cache_actual()])
    assert all([root_parent_second._is_cache_actual(), root_parent_first_parent_first._is_cache_actual(),
                root_parent_first_parent_second._is_cache_actual()])
    assert all([primary_node_first._is_cache_actual(), primary_node_second._is_cache_actual()])
