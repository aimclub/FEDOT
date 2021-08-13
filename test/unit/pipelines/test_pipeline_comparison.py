import itertools
from copy import deepcopy

import pytest

from fedot.core.optimisers.gp_comp.gp_operators import equivalent_subtree
from test.pipeline_manager import pipeline_first, pipeline_third, pipeline_second, pipeline_fourth


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
