import itertools
from copy import deepcopy

import pytest

from fedot.core.optimisers.gp_comp.gp_operators import equivalent_subtree
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline


def pipeline_first():
    #    RF
    #  |     \
    # RF     KNN
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
    #      RF
    #   |      \
    #  RF      KNN
    #  | \      |  \
    # LR RF   LR   LDA
    #    |  \
    #   KNN  LDA
    new_node = PipelineNode('rf')
    for model_type in ('knn', 'lda'):
        new_node.nodes_from.append(PipelineNode(model_type))
    pipeline = pipeline_first()
    pipeline.update_subtree(pipeline.root_node.nodes_from[0].nodes_from[1], new_node)

    return pipeline


def pipeline_third():
    #      RF
    #   /  |  \
    #  KNN LDA KNN
    root_of_tree = PipelineNode('rf')
    for model_type in ('knn', 'lda', 'knn'):
        root_of_tree.nodes_from.append(PipelineNode(model_type))
    pipeline = Pipeline()

    for node in root_of_tree.nodes_from:
        pipeline.add_node(node)
    pipeline.add_node(root_of_tree)

    return pipeline


def pipeline_fourth():
    #      RF
    #   |  \  \
    #  KNN  RF  KNN
    #      |  \
    #    KNN   KNN

    pipeline = pipeline_third()
    new_node = PipelineNode('rf')
    [new_node.nodes_from.append(PipelineNode('knn')) for _ in range(2)]
    pipeline.update_subtree(pipeline.root_node.nodes_from[1], new_node)

    return pipeline


@pytest.fixture()
def equality_cases():
    pairs = [[pipeline_first(), pipeline_first()], [pipeline_third(), pipeline_third()],
             [pipeline_fourth(), pipeline_fourth()]]

    # the following changes don't affect to pipelines equality:
    for node_num, type in enumerate(['knn', 'lda']):
        pairs[1][1].root_node.nodes_from[node_num].operation.operation_type = type

    for node_num in ((2, 1), (1, 2)):
        old_node = pairs[2][1].root_node.nodes_from[node_num[0]]
        new_node = deepcopy(pairs[2][0].root_node.nodes_from[node_num[1]])
        pairs[2][1].update_subtree(old_node, new_node)

    return pairs


@pytest.fixture()
def non_equality_cases():
    return list(itertools.combinations([pipeline_first(), pipeline_second(), pipeline_third()], 2))


@pytest.mark.parametrize('pipeline_fixture', ['equality_cases'])
def test_equality_cases(pipeline_fixture, request):
    list_pipelines_pairs = request.getfixturevalue(pipeline_fixture)
    for pair in list_pipelines_pairs:
        assert pair[0] == pair[1]
        assert pair[1] == pair[0]


@pytest.mark.parametrize('pipeline_fixture', ['non_equality_cases'])
def test_non_equality_cases(pipeline_fixture, request):
    list_pipelines_pairs = request.getfixturevalue(pipeline_fixture)
    for pair in list_pipelines_pairs:
        assert not pair[0] == pair[1]
        assert not pair[1] == pair[0]


def test_pipelines_equivalent_subtree():
    c_first = pipeline_first()
    c_second = pipeline_second()
    c_third = pipeline_third()

    similar_nodes_first_and_second = equivalent_subtree(c_first, c_second)
    assert len(similar_nodes_first_and_second) == 6

    similar_nodes_first_and_third = equivalent_subtree(c_first, c_third)
    assert not similar_nodes_first_and_third

    similar_nodes_second_and_third = equivalent_subtree(c_second, c_third)
    assert not similar_nodes_second_and_third

    similar_nodes_third = equivalent_subtree(c_third, c_third)
    assert len(similar_nodes_third) == 4
