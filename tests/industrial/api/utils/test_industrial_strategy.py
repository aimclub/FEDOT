from types import SimpleNamespace

import numpy as np

from fedot.industrial.api.utils.industrial_strategy import IndustrialStrategy
from fedot.industrial.core.repository.constanst_repository import FEDOT_TUNER_STRATEGY, FEDOT_TUNING_METRICS


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


def test_industrial_strategy_sampling_strategy_uses_base_input_without_mutation(monkeypatch):
    strategy = IndustrialStrategy(
        {'sampling_algorithm': 'CUR', 'sampling_range': [0.2, 0.5]},
        'sampling_strategy',
        {'timeout': 10, 'problem': 'classification'},
    )
    fitted_shapes = []

    class FakeFedot:
        def __init__(self, **config):
            self.config = config

        def fit(self, input_data):
            fitted_shapes.append(input_data.features.shape)
            return 'ok'

    def fake_sampler(tensor, target, sampling_rate):
        return f'decomposer:{sampling_rate}', tensor[:, :1], target[:1]

    base_input = SimpleNamespace(
        features=np.arange(12).reshape(2, 3, 2),
        target=np.arange(4).reshape(2, 2),
        idx=np.arange(2),
    )
    original_features = base_input.features.copy()
    original_target = base_input.target.copy()

    strategy.sampling_algorithm['CUR'] = fake_sampler
    monkeypatch.setattr('fedot.industrial.api.utils.industrial_strategy.Fedot', FakeFedot)

    strategy._sampling_strategy(base_input)

    assert np.array_equal(base_input.features, original_features)
    assert np.array_equal(base_input.target, original_target)
    assert fitted_shapes == [(2, 1, 2), (2, 1, 2)]
    assert sorted(strategy.solver.keys()) == ['CUR_sampling_rate_0.2', 'CUR_sampling_rate_0.5']
    assert sorted(strategy.sampler.keys()) == ['CUR_sampling_rate_0.2', 'CUR_sampling_rate_0.5']


def test_industrial_strategy_finetune_loop_uses_normalized_tuning_plan(monkeypatch):
    strategy = IndustrialStrategy({}, 'kernel_automl', {'timeout': 10, 'problem': 'classification'})
    captured = []

    monkeypatch.setattr(
        'fedot.industrial.api.utils.industrial_strategy.build_tuner',
        lambda api, model_to_tune, tuning_params, train_data, mode: captured.append((tuning_params, train_data, mode)) or ('pipeline_tuner', f'solver:{train_data}'),
    )

    result = strategy._finetune_loop(
        kernel_ensemble={'a': 'model-a', 'b': 'model-b'},
        kernel_data={'a': 'data-a', 'b': 'data-b'},
        tuning_params={'iterations': 3},
    )

    assert result == {'a': 'solver:data-a', 'b': 'solver:data-b'}
    assert captured[0][0]['iterations'] == 3
    assert captured[0][0]['metric'] == FEDOT_TUNING_METRICS['classification']
    assert captured[0][0]['tuner'] == FEDOT_TUNER_STRATEGY['simultaneous']
    assert captured[0][2] == 'head'
