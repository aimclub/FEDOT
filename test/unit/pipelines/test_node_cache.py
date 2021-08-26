import glob
import os

import pytest

from fedot.core.composer.cache import OperationsCache
from fedot.core.pipelines.node import SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from data.pipeline_manager import pipeline_fifth, pipeline_first, pipeline_second, pipeline_fourth, pipeline_third


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


def test_cache_actuality_after_model_change(data_setup):
    """The non-affected nodes has actual cache after changing the model"""

    cache = OperationsCache()

    pipeline = pipeline_first()
    train, _ = data_setup
    pipeline.fit(input_data=train)
    cache.save_pipeline(pipeline)
    new_node = SecondaryNode(operation_type='logit')
    pipeline.update_node(old_node=pipeline.root_node.nodes_from[0],
                         new_node=new_node)

    root_parent_first = pipeline.root_node.nodes_from[0]

    nodes_with_non_actual_cache = [pipeline.root_node, root_parent_first]
    nodes_with_actual_cache = [node for node in pipeline.nodes if node not in nodes_with_non_actual_cache]

    # non-affected nodes are actual
    assert all([cache.get(node) is not None for node in nodes_with_actual_cache])
    # affected nodes and their childs has no any actual cache
    assert all([cache.get(node) is None for node in nodes_with_non_actual_cache])


def test_cache_actuality_after_subtree_change_to_identical(data_setup):
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
    assert all([cache.get(node) is not None for node in nodes_with_actual_cache])
    # affected root node has no any actual cache
    assert cache.get(pipeline.root_node) is None


def test_cache_actuality_after_primary_node_changed_to_subtree(data_setup):
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
    assert all([cache.get(node) for node in nodes_with_actual_cache])
    # affected root nodes and their childs has no any actual cache
    assert not any([cache.get(node) for node in nodes_with_non_actual_cache])


def test_cache_historical_state_using(data_setup):
    cache = OperationsCache()
    train, _ = data_setup
    pipeline = pipeline_first()

    # pipeline fitted, model goes to cache
    pipeline.fit(input_data=train)
    cache.save_pipeline(pipeline)
    new_node = SecondaryNode(operation_type='logit')
    old_node = pipeline.root_node.nodes_from[0]

    # change child node to new one
    pipeline.update_node(old_node=old_node,
                         new_node=new_node)
    # cache is not actual
    assert not cache.get(pipeline.root_node)
    # fit modified pipeline
    pipeline.fit(input_data=train)
    cache.save_pipeline(pipeline)
    # cache is actual now
    assert cache.get(pipeline.root_node)

    # change node back
    pipeline.update_node(old_node=pipeline.root_node.nodes_from[0],
                         new_node=old_node)
    # cache is actual without new fitting,
    # because the cached model was saved after first fit
    assert cache.get(pipeline.root_node)


def test_multi_pipeline_caching_with_cache(data_setup):
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
    assert all([cache.get(node) for node in nodes_with_actual_cache])
    # the non-identical parts are still not fitted
    assert not any([cache.get(node) for node in nodes_with_non_actual_cache])

    # check the same case with another pipelines
    cache = OperationsCache()

    main_pipeline = pipeline_fourth()

    prev_pipeline_first = pipeline_third()
    prev_pipeline_second = pipeline_fifth()

    prev_pipeline_first.fit(input_data=train)
    cache.save_pipeline(prev_pipeline_first)
    prev_pipeline_second.fit(input_data=train)
    cache.save_pipeline(prev_pipeline_second)

    nodes_with_non_actual_cache = [main_pipeline.root_node, main_pipeline.root_node.nodes_from[1]]
    nodes_with_actual_cache = [child for child in main_pipeline.root_node.nodes_from[0].nodes_from]

    assert not any([cache.get(node) for node in nodes_with_non_actual_cache])
    assert all([cache.get(node) for node in nodes_with_actual_cache])

# TODO Add changed data case for cache
