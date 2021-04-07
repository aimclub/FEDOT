import glob
import os

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.cache import OperationsCache
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture()
def data_setup():
    task = Task(TaskTypesEnum.classification)
    predictors, response = load_breast_cancer(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    response = response[:100]
    predictors = predictors[:100]

    input_data = InputData(idx=np.arange(0, len(predictors)),
                           features=predictors,
                           target=response,
                           task=task,
                           data_type=DataTypesEnum.table)
    train_data, test_data = train_test_data_setup(data=input_data)
    train_data_x = train_data.features
    test_data_x = test_data.features
    train_data_y = train_data.target
    test_data_y = test_data.target

    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=np.arange(0, len(train_data_y)),
                           task=task, data_type=DataTypesEnum.table)
    test_data = InputData(features=test_data_x, target=test_data_y,
                          idx=np.arange(0, len(test_data_y)),
                          task=task, data_type=DataTypesEnum.table)
    return train_data, test_data


def create_func_delete_files(paths):
    """
    Create function to delete cache files after tests.
    """

    def wrapper():
        for path in paths:
            file_list = glob.glob(path)
            # Iterate over the list of filepaths & remove each file.
            for file_path in file_list:
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    return wrapper


@pytest.fixture(scope='session', autouse=True)
def preprocessing_files_before_and_after_tests(request):
    paths = ['*.bak', '*.dat', '*.dir']

    delete_files = create_func_delete_files(paths)
    delete_files()
    request.addfinalizer(delete_files)


def chain_first():
    #    XG
    #  |     \
    # XG      KNN
    # |  \    |  \
    # LR LDA LR  LDA
    chain = Chain()

    root_of_tree, root_child_first, root_child_second = \
        [SecondaryNode(model) for model in ('xgboost', 'xgboost', 'knn')]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in ('logit', 'lda'):
            new_node = PrimaryNode(requirement_model)
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
    new_node = SecondaryNode('dt')
    for model_type in ('knn', 'knn'):
        new_node.nodes_from.append(PrimaryNode(model_type))
    chain.replace_node_with_parents(chain.root_node.nodes_from[0], new_node)
    return chain


def chain_third():
    #    QDA
    #  |     \
    # RF     RF
    chain = Chain()
    new_node = SecondaryNode('qda')
    for model_type in ('rf', 'rf'):
        new_node.nodes_from.append(PrimaryNode(model_type))
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
    new_node = SecondaryNode('qda')
    for model_type in ('rf', 'rf'):
        new_node.nodes_from.append(PrimaryNode(model_type))
    chain.replace_node_with_parents(chain.root_node.nodes_from[0].nodes_from[1], new_node)
    new_node = SecondaryNode('knn')
    for model_type in ('knn', 'knn'):
        new_node.nodes_from.append(PrimaryNode(model_type))
    chain.replace_node_with_parents(chain.root_node.nodes_from[0].nodes_from[0], new_node)
    return chain


def chain_fifth():
    #    KNN
    #  |     \
    # XG      KNN
    # |  \    |  \
    # LR LDA KNN  KNN
    chain = chain_first()
    new_node = SecondaryNode('knn')
    chain.update_node(chain.root_node, new_node)
    new_node = PrimaryNode('knn')
    chain.update_node(chain.root_node.nodes_from[1].nodes_from[0], new_node)
    chain.update_node(chain.root_node.nodes_from[1].nodes_from[1], new_node)

    return chain


def test_cache_actuality_after_model_change(data_setup):
    """The non-affected nodes has actual cache after changing the model"""

    cache = OperationsCache()

    chain = chain_first()
    train, _ = data_setup
    chain.fit(input_data=train)
    cache.save_chain(chain)
    new_node = SecondaryNode(operation_type='logit')
    chain.update_node(old_node=chain.root_node.nodes_from[0],
                      new_node=new_node)

    root_parent_first = chain.root_node.nodes_from[0]

    nodes_with_non_actual_cache = [chain.root_node, root_parent_first]
    nodes_with_actual_cache = [node for node in chain.nodes if node not in nodes_with_non_actual_cache]

    # non-affected nodes are actual
    assert all([cache.get(node) is not None for node in nodes_with_actual_cache])
    # affected nodes and their childs has no any actual cache
    assert all([cache.get(node) is None for node in nodes_with_non_actual_cache])


def test_cache_actuality_after_subtree_change_to_identical(data_setup):
    """The non-affected nodes has actual cache after changing the subtree to other pre-fitted subtree"""
    cache = OperationsCache()
    train, _ = data_setup
    chain = chain_first()
    other_chain = chain_second()
    chain.fit(input_data=train)
    cache.save_chain(chain)
    other_chain.fit(input_data=train)
    cache.save_chain(Chain(other_chain.root_node.nodes_from[0]))

    chain.replace_node_with_parents(chain.root_node.nodes_from[0],
                                    other_chain.root_node.nodes_from[0])

    nodes_with_actual_cache = [node for node in chain.nodes if node not in [chain.root_node]]

    # non-affected nodes of initial chain and fitted nodes of new subtree are actual
    assert all([cache.get(node) is not None for node in nodes_with_actual_cache])
    # affected root node has no any actual cache
    assert cache.get(chain.root_node) is None


def test_cache_actuality_after_primary_node_changed_to_subtree(data_setup):
    """ The non-affected nodes has actual cache after changing the primary node to pre-fitted subtree"""
    cache = OperationsCache()
    train, _ = data_setup
    chain = chain_first()
    other_chain = chain_second()
    chain.fit(input_data=train)
    cache.save_chain(chain)
    other_chain.fit(input_data=train)
    chain.replace_node_with_parents(chain.root_node.nodes_from[0].nodes_from[0],
                                    other_chain.root_node.nodes_from[0])
    cache.save_chain(Chain(other_chain.root_node.nodes_from[0]))
    root_parent_first = chain.root_node.nodes_from[0]

    nodes_with_non_actual_cache = [chain.root_node, root_parent_first]
    nodes_with_actual_cache = [node for node in chain.nodes if node not in nodes_with_non_actual_cache]

    # non-affected nodes of initial chain and fitted nodes of new subtree are actual
    assert all([cache.get(node) for node in nodes_with_actual_cache])
    # affected root nodes and their childs has no any actual cache
    assert not any([cache.get(node) for node in nodes_with_non_actual_cache])


def test_cache_historical_state_using(data_setup):
    cache = OperationsCache()
    train, _ = data_setup
    chain = chain_first()

    # chain fitted, model goes to cache
    chain.fit(input_data=train)
    cache.save_chain(chain)
    new_node = SecondaryNode(operation_type='logit')
    old_node = chain.root_node.nodes_from[0]

    # change child node to new one
    chain.update_node(old_node=old_node,
                      new_node=new_node)
    # cache is not actual
    assert not cache.get(chain.root_node)
    # fit modified chain
    chain.fit(input_data=train)
    cache.save_chain(chain)
    # cache is actual now
    assert cache.get(chain.root_node)

    # change node back
    chain.update_node(old_node=chain.root_node.nodes_from[0],
                      new_node=old_node)
    # cache is actual without new fitting,
    # because the cached model was saved after first fit
    assert cache.get(chain.root_node)


def test_multi_chain_caching_with_cache(data_setup):
    train, _ = data_setup
    cache = OperationsCache()

    main_chain = chain_second()
    other_chain = chain_first()

    # fit other_chain that contains the parts identical to main_chain
    other_chain.fit(input_data=train)
    cache.save_chain(other_chain)

    nodes_with_non_actual_cache = [main_chain.root_node, main_chain.root_node.nodes_from[0]] + \
                                  [_ for _ in main_chain.root_node.nodes_from[0].nodes_from]
    nodes_with_actual_cache = [node for node in main_chain.nodes if node not in nodes_with_non_actual_cache]

    # check that using of other_chain make identical of the main_chain fitted,
    # despite the main_chain.fit() was not called
    assert all([cache.get(node) for node in nodes_with_actual_cache])
    # the non-identical parts are still not fitted
    assert not any([cache.get(node) for node in nodes_with_non_actual_cache])

    # check the same case with another chains
    cache = OperationsCache()

    main_chain = chain_fourth()

    prev_chain_first = chain_third()
    prev_chain_second = chain_fifth()

    prev_chain_first.fit(input_data=train)
    cache.save_chain(prev_chain_first)
    prev_chain_second.fit(input_data=train)
    cache.save_chain(prev_chain_second)

    nodes_with_non_actual_cache = [main_chain.root_node, main_chain.root_node.nodes_from[1]]
    nodes_with_actual_cache = [child for child in main_chain.root_node.nodes_from[0].nodes_from]

    assert not any([cache.get(node) for node in nodes_with_non_actual_cache])
    assert all([cache.get(node) for node in nodes_with_actual_cache])

# TODO Add changed data case for cache
