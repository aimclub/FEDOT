from types import SimpleNamespace

import numpy as np

from fedot.industrial.api.utils.industrial_strategy import IndustrialStrategy


def test_industrial_strategy_fit_and_predict_use_rule_based_dispatch(monkeypatch):
    strategy = IndustrialStrategy({}, 'federated_automl', {'timeout': 10, 'problem': 'classification'})
    captured = {}

    monkeypatch.setattr(
        'fedot.industrial.api.utils.industrial_strategy.resolve_industrial_strategy_dispatch',
        lambda strategy_name: SimpleNamespace(
            fit_method_name='_fit_rule_method',
            predict_method_name='_predict_rule_method',
        ),
    )

    strategy._fit_rule_method = lambda input_data: captured.update({'fit_input': input_data, 'solver': 'solver'}) or setattr(strategy, 'solver', 'solver')
    strategy._predict_rule_method = lambda input_data, mode: {'input_data': input_data, 'mode': mode}

    fit_result = strategy.fit('train-data')
    predict_result = strategy.predict('test-data', 'labels')

    assert fit_result == 'solver'
    assert captured['fit_input'] == 'train-data'
    assert predict_result == {'input_data': 'test-data', 'mode': 'labels'}


def test_industrial_strategy_sampling_predict_keeps_full_feature_space_for_non_cur():
    strategy = IndustrialStrategy({'sampling_algorithm': 'Random'}, 'sampling_strategy', {'timeout': 10, 'problem': 'classification'})
    captured = {}

    class FakeSolver:
        def predict_proba(self, input_data):
            captured['features_shape'] = input_data.features.shape
            return 'probs'

    strategy.solver = {'sample': FakeSolver()}
    strategy.sampler = {'sample': object()}
    input_data = SimpleNamespace(features=np.arange(24).reshape(4, 3, 2))

    result = strategy._sampling_predict(input_data, mode='probs')

    assert result == {'sample': 'probs'}
    assert captured['features_shape'] == (4, 3, 2)
