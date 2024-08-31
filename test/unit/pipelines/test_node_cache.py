import glob
import os

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from fedot.core.caching.operations_cache import OperationsCache
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture()
def data_setup():
    task = Task(TaskTypesEnum.classification)
    predictors, response = load_breast_cancer(return_X_y=True)
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


@pytest.fixture
def cache_cleanup():
    OperationsCache().reset()
    yield
    OperationsCache().reset()


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


def pipeline_first():
    #    XG
    #  |     \
    # XG      KNN
    # |  \    |  \
    # LR LDA LR  LDA
    pipeline = Pipeline()

    root_of_tree, root_child_first, root_child_second = \
        [PipelineNode(model) for model in ('rf', 'rf', 'knn')]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in ('logit', 'lda'):
            new_node = PipelineNode(requirement_model)
            root_node_child.nodes_from.append(new_node)
            pipeline.add_node(new_node)
        pipeline.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    pipeline.add_node(root_of_tree)
    return pipeline


def pipeline_second():
    #    XG
    #  |     \
    # DT      KNN
    # |  \    |  \
    # KNN KNN LR  LDA
    pipeline = pipeline_first()
    new_node = PipelineNode('dt')
    for model_type in ('knn', 'knn'):
        new_node.nodes_from.append(PipelineNode(model_type))
    pipeline.update_subtree(pipeline.root_node.nodes_from[0], new_node)
    return pipeline


def pipeline_third():
    #    QDA
    #  |     \
    # RF     RF
    pipeline = Pipeline()
    new_node = PipelineNode('qda')
    for model_type in ('rf', 'rf'):
        new_node.nodes_from.append(PipelineNode(model_type))
    pipeline.add_node(new_node)
    [pipeline.add_node(node_from) for node_from in new_node.nodes_from]
    return pipeline


def pipeline_fourth():
    #          XG
    #      |         \
    #     XG          KNN
    #   |    \        |  \
    # QDA     KNN     LR  LDA
    # |  \    |    \
    # RF  RF  KNN KNN
    pipeline = pipeline_first()
    new_node = PipelineNode('qda')
    for model_type in ('rf', 'rf'):
        new_node.nodes_from.append(PipelineNode(model_type))
    pipeline.update_subtree(pipeline.root_node.nodes_from[0].nodes_from[1], new_node)
    new_node = PipelineNode('knn')
    for model_type in ('knn', 'knn'):
        new_node.nodes_from.append(PipelineNode(model_type))
    pipeline.update_subtree(pipeline.root_node.nodes_from[0].nodes_from[0], new_node)
    return pipeline


def pipeline_fifth():
    #    KNN
    #  |     \
    # XG      KNN
    # |  \    |  \
    # LR LDA KNN  KNN
    pipeline = pipeline_first()
    new_node = PipelineNode('knn')
    pipeline.update_node(pipeline.root_node, new_node)
    new_node1 = PipelineNode('knn')
    new_node2 = PipelineNode('knn')
    pipeline.update_node(pipeline.root_node.nodes_from[1].nodes_from[0], new_node1)
    pipeline.update_node(pipeline.root_node.nodes_from[1].nodes_from[1], new_node2)

    return pipeline


def test_cache_actuality_after_model_change(data_setup, cache_cleanup):
    """The non-affected nodes has actual cache after changing the model"""

    cache = OperationsCache()

    pipeline = pipeline_first()
    train, _ = data_setup
    pipeline.fit(input_data=train)
    cache.save_pipeline(pipeline)
    new_node = PipelineNode(operation_type='logit')
    pipeline.update_node(old_node=pipeline.root_node.nodes_from[0],
                         new_node=new_node)

    root_parent_first = pipeline.root_node.nodes_from[0]

    nodes_with_non_actual_cache = [pipeline.root_node, root_parent_first]
    nodes_with_actual_cache = [node for node in pipeline.nodes if node not in nodes_with_non_actual_cache]

    # non-affected nodes are actual
    cache.try_load_nodes(nodes_with_actual_cache)
    assert all(node.fitted_operation is not None for node in nodes_with_actual_cache)
    # affected nodes and their childs has no any actual cache
    cache.try_load_nodes(nodes_with_non_actual_cache)
    assert all(node.fitted_operation is None for node in nodes_with_non_actual_cache)


def test_cache_actuality_after_subtree_change_to_identical(data_setup, cache_cleanup):
    """The non-affected nodes has actual cache after changing the subtree to other pre-fitted subtree"""
    cache = OperationsCache()
    train, _ = data_setup
    pipeline = pipeline_first()
    other_pipeline = pipeline_second()
    pipeline.fit(input_data=train)
    cache.save_pipeline(pipeline)
    other_pipeline.fit(input_data=train)
    cache.save_pipeline(Pipeline(other_pipeline.root_node.nodes_from[0]))

    pipeline.update_subtree(pipeline.root_node.nodes_from[0],
                            other_pipeline.root_node.nodes_from[0])

    nodes_with_actual_cache = [node for node in pipeline.nodes if node not in [pipeline.root_node]]

    # non-affected nodes of initial pipeline and fitted nodes of new subtree are actual
    cache.try_load_nodes(nodes_with_actual_cache)
    assert all(node.fitted_operation is not None for node in nodes_with_actual_cache)
    # affected root node has no any actual cache
    cache.try_load_nodes(pipeline.root_node)
    assert pipeline.root_node.fitted_operation is None


def test_cache_actuality_after_primary_node_changed_to_subtree(data_setup, cache_cleanup):
    """ The non-affected nodes has actual cache after changing the primary node to pre-fitted subtree"""
    cache = OperationsCache()
    train, _ = data_setup
    pipeline = pipeline_first()
    other_pipeline = pipeline_second()
    pipeline.fit(input_data=train)
    cache.save_pipeline(pipeline)
    other_pipeline.fit(input_data=train)
    pipeline.update_subtree(pipeline.root_node.nodes_from[0].nodes_from[0],
                            other_pipeline.root_node.nodes_from[0])
    cache.save_pipeline(Pipeline(other_pipeline.root_node.nodes_from[0]))
    root_parent_first = pipeline.root_node.nodes_from[0]

    nodes_with_non_actual_cache = [pipeline.root_node, root_parent_first]
    nodes_with_actual_cache = [node for node in pipeline.nodes if node not in nodes_with_non_actual_cache]

    # non-affected nodes of initial pipeline and fitted nodes of new subtree are actual
    cache.try_load_nodes(nodes_with_actual_cache)
    assert all(node.fitted_operation is not None for node in nodes_with_actual_cache)
    # affected root nodes and their childs has no any actual cache
    cache.try_load_nodes(nodes_with_non_actual_cache)
    assert all(node.fitted_operation is None for node in nodes_with_non_actual_cache)


def test_cache_historical_state_using_with_cv(data_setup, cache_cleanup):
    cv_fold = 1
    cache = OperationsCache()
    train, _ = data_setup
    pipeline = pipeline_first()

    # pipeline fitted, model goes to cache
    pipeline.fit(input_data=train)
    cache.save_pipeline(pipeline, fold_id=cv_fold)
    new_node = PipelineNode(operation_type='logit')
    old_node = pipeline.root_node.nodes_from[0]

    # change child node to new one
    pipeline.update_node(old_node=old_node,
                         new_node=new_node)
    # cache is not actual
    cache.try_load_nodes(pipeline.root_node)
    assert pipeline.root_node.fitted_operation is None
    # fit modified pipeline
    pipeline.fit(input_data=train)
    cache.save_pipeline(pipeline, fold_id=cv_fold)
    # cache is actual now
    cache.try_load_nodes(pipeline.root_node, fold_id=cv_fold)
    assert pipeline.root_node.fitted_operation is not None

    # change node back
    pipeline.update_node(old_node=pipeline.root_node.nodes_from[0],
                         new_node=old_node)
    # cache is actual without new fitting,
    # because the cached model was saved after first fit
    cache.try_load_nodes(pipeline.root_node, fold_id=cv_fold)
    assert pipeline.root_node.fitted_operation is not None


def test_multi_pipeline_caching_with_cache(data_setup, cache_cleanup):
    train, _ = data_setup
    cache = OperationsCache()

    main_pipeline = pipeline_second()
    other_pipeline = pipeline_first()

    # fit other_pipeline that contains the parts identical to main_pipeline
    other_pipeline.fit(input_data=train)
    cache.save_pipeline(other_pipeline)

    nodes_with_non_actual_cache = [main_pipeline.root_node, main_pipeline.root_node.nodes_from[0]] + \
                                  [_ for _ in main_pipeline.root_node.nodes_from[0].nodes_from]
    nodes_with_actual_cache = [node for node in main_pipeline.nodes if node not in nodes_with_non_actual_cache]

    # check that using of other_pipeline make identical of the main_pipeline fitted,
    # despite the main_pipeline.fit() was not called
    cache.try_load_nodes(nodes_with_actual_cache)
    assert all(node.fitted_operation is not None for node in nodes_with_actual_cache)
    # the non-identical parts are still not fitted
    cache.try_load_nodes(nodes_with_non_actual_cache)
    assert all(node.fitted_operation is None for node in nodes_with_non_actual_cache)

    # check the same case with another pipelines
    cache.reset()

    main_pipeline = pipeline_fourth()

    prev_pipeline_first = pipeline_third()
    prev_pipeline_second = pipeline_fifth()

    prev_pipeline_first.fit(input_data=train)
    cache.save_pipeline(prev_pipeline_first)
    prev_pipeline_second.fit(input_data=train)
    cache.save_pipeline(prev_pipeline_second)

    nodes_with_non_actual_cache = [main_pipeline.root_node, main_pipeline.root_node.nodes_from[1]]
    nodes_with_actual_cache = [child for child in main_pipeline.root_node.nodes_from[0].nodes_from]

    cache.try_load_nodes(nodes_with_non_actual_cache)
    assert all(node.fitted_operation is None for node in nodes_with_non_actual_cache)
    cache.try_load_nodes(nodes_with_actual_cache)
    assert all(node.fitted_operation is not None for node in nodes_with_actual_cache)

# TODO Add changed data case for cache
