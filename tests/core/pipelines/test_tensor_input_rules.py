import pytest

from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.tensor_input_rules import resolve_decompose_parents, resolve_tensor_target_source


def correction_graph():
    features = PipelineNode('scaling')
    baseline = PipelineNode('linear', nodes_from=[features])
    residual = PipelineNode('decompose', nodes_from=[features, baseline])
    correction = PipelineNode('ridge', nodes_from=[residual])
    return features, baseline, residual, correction


@pytest.mark.parametrize('reverse', [False, True])
def test_main_target_wins_independently_of_parent_order(reverse):
    _, baseline, _, correction = correction_graph()
    parents = [correction, baseline] if reverse else [baseline, correction]
    assert parents[resolve_tensor_target_source(parents)] is baseline


def test_labels_can_live_in_one_main_branch_but_do_not_promote_an_auxiliary_objective():
    features, baseline, _, correction = correction_graph()
    parents = [features, correction, baseline]
    assert resolve_tensor_target_source(parents, [False, True, True]) == 2
    assert resolve_tensor_target_source(parents, [False, True, False]) == 0


def test_shallowest_auxiliary_wins_and_ties_keep_supplied_order():
    _, _, residual, correction = correction_graph()
    second_residual = PipelineNode('decompose', nodes_from=[residual, correction])
    deeper = PipelineNode('linear', nodes_from=[second_residual])
    assert resolve_tensor_target_source([deeper, correction]) == 1
    sibling = PipelineNode('linear', nodes_from=[residual])
    assert resolve_tensor_target_source([correction, sibling]) == 0
    assert resolve_tensor_target_source([sibling, correction]) == 0


def test_merging_auxiliary_forks_does_not_make_their_target_main():
    _, baseline, residual, correction = correction_graph()
    sibling = PipelineNode('linear', nodes_from=[residual])
    ensemble = PipelineNode('ridge', nodes_from=[correction, sibling])
    assert resolve_tensor_target_source([ensemble, baseline]) == 1


def test_graph_mutation_is_not_hidden_by_persisted_lineage():
    features, baseline, _, correction = correction_graph()
    assert resolve_tensor_target_source([correction, baseline]) == 1
    correction.nodes_from = [features]
    assert resolve_tensor_target_source([correction, baseline]) == 0


def test_decompose_alias_is_a_target_boundary():
    _, baseline, _, _ = correction_graph()
    residual = PipelineNode('decompose/correction', nodes_from=[PipelineNode('scaling'), baseline])
    assert resolve_tensor_target_source([residual, baseline]) == 1


@pytest.mark.parametrize('reverse', [False, True])
def test_explicit_parent_roles_win_even_if_feature_branch_is_deeper(reverse):
    model = PipelineNode('linear')
    features = PipelineNode('scaling', nodes_from=[PipelineNode('normalization')])
    parents = [features, model] if reverse else [model, features]
    prediction, data = resolve_decompose_parents(parents)
    assert parents[prediction] is model
    assert parents[data] is features


def test_two_models_use_depth_and_stable_tie_break():
    shallow = PipelineNode('linear')
    deep = PipelineNode('ridge', nodes_from=[shallow])
    assert resolve_decompose_parents([shallow, deep]) == (1, 0)
    assert resolve_decompose_parents([deep, shallow]) == (0, 1)
    assert resolve_decompose_parents([shallow, PipelineNode('ridge')]) == (0, 1)


@pytest.mark.parametrize('names', [[], ['linear'], ['linear', 'ridge', 'linear'], ['scaling', 'normalization']])
def test_invalid_parent_roles_or_counts_are_rejected(names):
    with pytest.raises(ValueError, match='exactly two|Model parent'):
        resolve_decompose_parents([PipelineNode(name) for name in names])
