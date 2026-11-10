from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import pytest

from examples.advanced import fixed_multiclass_graph_probe as probe


def test_component_probe_changes_one_block_and_saves_predictions(tmp_path, monkeypatch):
    data_path = tmp_path / 'train.npz'
    np.savez(data_path, features=np.zeros((90, 2)), target=np.arange(90) % 3, task_id=7593, fold=3)
    baseline = {name: {'sentinel': 1} for name in ('scaling', 'catboost', 'xgboost', 'lgbm', 'logit')}
    candidate = {name: {'sentinel': 2} for name in baseline}
    monkeypatch.setattr(probe, 'final_parameters', lambda path: baseline if path == 'base' else candidate)
    observed = []
    monkeypatch.setattr(probe.Pipeline, 'fit', lambda self, data, **kwargs: observed.append(
        {node.name: dict(node.parameters) for node in self.nodes}))
    monkeypatch.setattr(probe.Pipeline, 'predict', lambda self, data, **kwargs: SimpleNamespace(
        predict=np.full((len(data.target), 3), 1 / 3)))
    args = SimpleNamespace(data=data_path, output=tmp_path / 'result.json',
                           baseline_log='base', candidate_log='new',
                           component_ablation=True, save_predictions=True)
    probe.main(args)
    assert len(observed) == 6
    for parameters, operation in zip(observed[2:], ('catboost', 'xgboost', 'lgbm', 'logit')):
        changed = [name for name in baseline if parameters[name]['sentinel'] != baseline[name]['sentinel']]
        assert changed == [operation]
        assert parameters['lgbm']['min_child_weight'] == 0.001
    predictions = list(tmp_path.glob('result-cell*.npz'))
    assert len(predictions) == 6
    with np.load(predictions[0]) as saved:
        assert saved['probabilities'].shape == (18, 3)
        assert saved['truth'].shape == (18,)
        assert saved['classes'].tolist() == [0, 1, 2]
    assert baseline['lgbm'] == {'sentinel': 1}


def lgbm_parameters():
    fields = ('learning_rate', 'num_leaves', 'min_data_in_leaf',
              'colsample_bytree', 'reg_alpha', 'reg_lambda', 'subsample')
    baseline = {name: {'sentinel': 1} for name in ('scaling', 'catboost', 'xgboost', 'lgbm', 'logit')}
    baseline['lgbm'].update({field: 1 for field in fields})
    candidate = deepcopy(baseline)
    candidate['lgbm'].update({field: 2 for field in fields})
    return baseline, candidate, fields


def test_lgbm_probe_changes_exactly_one_parameter_without_mutation():
    baseline, candidate, fields = lgbm_parameters()
    before = deepcopy((baseline, candidate))
    variants = probe.parameter_variants(baseline, candidate, lgbm_ablation=True)
    assert len(variants) == 9
    assert variants[0] == ('baseline', baseline)
    assert variants[1] == ('swap_lgbm', candidate)
    for (label, parameters), field in zip(variants[2:], fields):
        assert label == 'lgbm_' + field
        assert [key for key in baseline['lgbm'] if parameters['lgbm'][key] != baseline['lgbm'][key]] == [field]
        assert all(parameters[name] == baseline[name] for name in baseline if name != 'lgbm')
    variants[2][1]['lgbm']['sentinel'] = 99
    assert (baseline, candidate) == before
    assert variants[0][1]['lgbm']['sentinel'] == 1


def test_lgbm_probe_rejects_unexpected_parameter_differences():
    baseline, candidate, _ = lgbm_parameters()
    candidate['lgbm']['unexpected'] = 2
    with pytest.raises(ValueError, match='Unexpected LightGBM'):
        probe.parameter_variants(baseline, candidate, lgbm_ablation=True)
    with pytest.raises(ValueError, match='only one'):
        probe.parameter_variants(baseline, candidate, component_ablation=True, lgbm_ablation=True)
