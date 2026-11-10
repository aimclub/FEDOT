from types import SimpleNamespace

import numpy as np

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
