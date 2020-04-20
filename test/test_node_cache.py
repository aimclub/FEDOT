import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData, split_train_test
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.task_types import MachineLearningTasksEnum


@pytest.fixture()
def data_setup():
    task_type = MachineLearningTasksEnum.classification
    predictors, response = load_breast_cancer(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    response = response[:100]
    predictors = predictors[:100]
    train_data_x, test_data_x = split_train_test(predictors)
    train_data_y, test_data_y = split_train_test(response)
    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=np.arange(0, len(train_data_y)),
                           task_type=task_type)
    test_data = InputData(features=test_data_x, target=test_data_y,
                          idx=np.arange(0, len(test_data_y)),
                          task_type=task_type)
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
    #    XG
    #  |     \
    # DT      KNN
    # |  \    |  \
    # KNN KNN LR  LDA
    chain = chain_first()
    new_node = NodeGenerator.secondary_node(ModelTypesIdsEnum.dt)
    for model_type in (ModelTypesIdsEnum.knn, ModelTypesIdsEnum.knn):
        new_node.nodes_from.append(NodeGenerator.primary_node(model_type))
    chain.replace_node_with_parents(chain.root_node.nodes_from[0], new_node)
    return chain


def chain_third():
    #    QDA
    #  |     \
    # RF     RF
    chain = Chain()
    new_node = NodeGenerator.secondary_node(ModelTypesIdsEnum.qda)
    for model_type in (ModelTypesIdsEnum.rf, ModelTypesIdsEnum.rf):
        new_node.nodes_from.append(NodeGenerator.primary_node(model_type))
    chain.add_node(new_node)
    [chain.add_node(node_from) for node_from in new_node.nodes_from]
    return chain


def chain_fourth():
    #          XG
    #      |         \
    #     XG          KNN
    #   |    \        |  \
    # QDA     KNN     LR  LDA
    # |  \    |    \
    # RF  RF  KNN KNN
    chain = chain_first()
    new_node = NodeGenerator.secondary_node(ModelTypesIdsEnum.qda)
    for model_type in (ModelTypesIdsEnum.rf, ModelTypesIdsEnum.rf):
        new_node.nodes_from.append(NodeGenerator.primary_node(model_type))
    chain.replace_node_with_parents(chain.root_node.nodes_from[0].nodes_from[1], new_node)
    new_node = NodeGenerator.secondary_node(ModelTypesIdsEnum.knn)
    for model_type in (ModelTypesIdsEnum.knn, ModelTypesIdsEnum.knn):
        new_node.nodes_from.append(NodeGenerator.primary_node(model_type))
    chain.replace_node_with_parents(chain.root_node.nodes_from[0].nodes_from[0], new_node)
    return chain


def chain_fifth():
    #    KNN
    #  |     \
    # XG      KNN
    # |  \    |  \
    # LR LDA KNN  KNN
    chain = chain_first()
    new_node = NodeGenerator.secondary_node(ModelTypesIdsEnum.knn)
    chain.update_node(chain.root_node, new_node)
    new_node = NodeGenerator.primary_node(ModelTypesIdsEnum.knn)
    chain.update_node(chain.root_node.nodes_from[1].nodes_from[0], new_node)
    chain.update_node(chain.root_node.nodes_from[1].nodes_from[1], new_node)

    return chain


def test_cache_model_changed(data_setup):
    """Changing the model in one of the tree node"""
    chain = chain_first()
    train, _ = data_setup
    chain.fit(input_data=train)
    new_node = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit)
    chain.update_node(old_node=chain.root_node.nodes_from[0],
                      new_node=new_node)

    root_parent_first = chain.root_node.nodes_from[0]

    nodes_with_non_actual_cache = [chain.root_node, root_parent_first]
    nodes_with_actual_cache = [node for node in chain.nodes if node not in nodes_with_non_actual_cache]

    assert not any([node.cache.actual_cached_model for node in nodes_with_non_actual_cache])
    assert all([node.cache.actual_cached_model for node in nodes_with_actual_cache])


def test_cache_subtree_changed(data_setup):
    """The subtree in source tree changed to other previously trained subtree"""
    train, _ = data_setup
    chain = chain_first()
    other_chain = chain_second()
    chain.fit(input_data=train)
    other_chain.fit(input_data=train)
    chain.replace_node_with_parents(chain.root_node.nodes_from[0],
                                    other_chain.root_node.nodes_from[0])

    nodes_with_actual_cache = [node for node in chain.nodes if node not in [chain.root_node]]

    assert not chain.root_node.cache.actual_cached_model
    assert all([node.cache.actual_cached_model for node in nodes_with_actual_cache])


def test_cache_primary_node_changed_to_subtree(data_setup):
    """The primary node in source tree changed to other previously trained subtree"""
    train, _ = data_setup
    chain = chain_first()
    other_chain = chain_second()
    chain.fit(input_data=train)
    other_chain.fit(input_data=train)
    chain.replace_node_with_parents(chain.root_node.nodes_from[0].nodes_from[0],
                                    other_chain.root_node.nodes_from[0])

    root_parent_first = chain.root_node.nodes_from[0]

    nodes_with_non_actual_cache = [chain.root_node, root_parent_first]
    nodes_with_actual_cache = [node for node in chain.nodes if node not in nodes_with_non_actual_cache]

    assert not any([node.cache.actual_cached_model for node in nodes_with_non_actual_cache])
    assert all([node.cache.actual_cached_model for node in nodes_with_actual_cache])


def test_cache_dictionary(data_setup):
    train, _ = data_setup
    chain = chain_first()
    chain.fit(input_data=train)
    new_node = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit)
    old_node = chain.root_node.nodes_from[0]

    # change child node
    chain.update_node(old_node=old_node,
                      new_node=new_node)
    assert not chain.root_node.cache.actual_cached_model

    # change back
    chain.update_node(old_node=chain.root_node.nodes_from[0],
                      new_node=old_node)
    assert chain.root_node.cache.actual_cached_model


def test_multi_chain_caching_global_cache(data_setup):
    _multi_chain_caching(data_setup, is_global_cache=True)


def test_multi_chain_caching_local_cache(data_setup):
    _multi_chain_caching(data_setup, is_global_cache=False)


def test_no_multi_chain_caching(data_setup):
    _multi_chain_caching(data_setup, try_multichain=False)


def _multi_chain_caching(data_setup, try_multichain=True, is_global_cache=False):
    train, _ = data_setup
    chain = chain_second()
    other_chain = chain_first()

    if try_multichain and is_global_cache:
        shared_cache = {}
        chain.shared_cache = shared_cache
        other_chain.shared_cache = shared_cache

    other_chain.fit(input_data=train)

    if try_multichain and not is_global_cache:
        chain.import_cache(other_chain)

    nodes_with_non_actual_cache = [chain.root_node, chain.root_node.nodes_from[0]]+[child for child in chain.root_node.nodes_from[0].nodes_from]
    nodes_with_actual_cache = [node for node in chain.nodes if node not in nodes_with_non_actual_cache]

    if try_multichain:
        assert not any([node.cache.actual_cached_model for node in nodes_with_non_actual_cache])
        assert all([node.cache.actual_cached_model for node in nodes_with_actual_cache])
    else:
        assert not any([node.cache.actual_cached_model for node in chain.nodes])

    chain = chain_fourth()
    prev_chain_first = chain_third()
    prev_chain_second = chain_fifth()

    if try_multichain and is_global_cache:
        shared_cache = {}
        chain.shared_cache = shared_cache
        prev_chain_first.shared_cache = shared_cache
        prev_chain_second.shared_cache = shared_cache

    prev_chain_first.fit(input_data=train)
    prev_chain_second.fit(input_data=train)

    if try_multichain and not is_global_cache:
        chain.import_cache(prev_chain_first)
        chain.import_cache(prev_chain_second)

    nodes_with_non_actual_cache = [chain.root_node, chain.root_node.nodes_from[1]]
    nodes_with_actual_cache = [child for child in chain.root_node.nodes_from[0].nodes_from]
    if try_multichain:
        assert not any([node.cache.actual_cached_model for node in nodes_with_non_actual_cache])
        assert all([node.cache.actual_cached_model for node in nodes_with_actual_cache])
    else:
        assert not any([node.cache.actual_cached_model for node in chain.nodes])
