import numpy as np

from fedot.api.api_utils.input_analyser import InputAnalyser
from fedot.core.data.input_data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _make_input_data():
    features = np.array([
        ['a', 'x', 1],
        ['b', 'y', 2],
        ['c', 'z', 3],
        ['d', 'q', 4],
    ], dtype=object)
    target = np.array([0, 1, 0, 1])
    return InputData(
        idx=np.arange(features.shape[0]),
        features=features,
        target=target,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )


def test_input_analyser_give_recommendations_uses_pure_bundle_rules(monkeypatch):
    captured = {}

    def fake_build_recommendation_bundle(**kwargs):
        captured['safe_mode'] = kwargs['safe_mode']
        captured['input_params'] = kwargs['input_params']
        return type('Bundle', (), {'data': {'cut': {'border': 2}}, 'params': {'preset': 'fast_train'}})()

    monkeypatch.setattr(
        'fedot.api.api_utils.input_analyser.build_recommendation_bundle',
        fake_build_recommendation_bundle)

    analyser = InputAnalyser(safe_mode=True)
    data_recommendations, params_recommendations = analyser.give_recommendations(
        _make_input_data(),
        input_params={'use_meta_rules': True},
    )

    assert captured['safe_mode'] is True
    assert captured['input_params'] == {'use_meta_rules': True}
    assert data_recommendations == {'cut': {'border': 2}}
    assert params_recommendations == {'preset': 'fast_train'}


def test_input_analyser_control_helpers_delegate_to_rules():
    analyser = InputAnalyser(safe_mode=True)
    analyser.max_size = 8
    analyser.max_cat_cardinality = 5

    input_data = _make_input_data()

    is_cut_needed, border = analyser.control_size(input_data)

    assert is_cut_needed is True
    assert border == 2
    assert analyser.control_categorical(input_data) is True
