from types import SimpleNamespace

import numpy as np

from fedot.api.api_utils.recommendation_rules import (
    RecommendationLimits,
    build_recommendation_bundle,
    build_safe_data_recommendations,
    collect_meta_rule_recommendations,
    estimate_categorical_cardinality,
    estimate_size_cut_border,
    merge_parameter_recommendations,
    should_apply_meta_rules,
    should_use_label_encoding,
)
from fedot.core.repository.dataset_types import DataTypesEnum


class _FakeLog:
    def info(self, message):
        return message


def _fake_categorical_detector(_table):
    return [0, 1], [2]


def test_estimate_size_cut_border_returns_border_only_for_large_tables():
    input_data = SimpleNamespace(
        data_type=DataTypesEnum.table, features=np.zeros((10, 4)))

    assert estimate_size_cut_border(input_data, max_size=30) == 7
    assert estimate_size_cut_border(input_data, max_size=100) is None


def test_estimate_categorical_cardinality_and_label_encoding_decision():
    input_data = SimpleNamespace(
        data_type=DataTypesEnum.table,
        features=np.array([
            ['a', 'x', 1],
            ['b', 'y', 2],
            ['c', 'z', 3],
        ], dtype=object),
    )

    cardinality = estimate_categorical_cardinality(
        input_data, _fake_categorical_detector)
    assert cardinality == 6
    assert should_use_label_encoding(
        input_data, max_cat_cardinality=5, categorical_detector=_fake_categorical_detector)


def test_build_safe_data_recommendations_is_empty_when_safe_mode_disabled():
    input_data = SimpleNamespace(
        data_type=DataTypesEnum.table, features=np.zeros((10, 4)))

    recommendations = build_safe_data_recommendations(
        input_data=input_data,
        safe_mode=False,
        limits=RecommendationLimits(max_size=10, max_cat_cardinality=1),
        categorical_detector=_fake_categorical_detector,
    )

    assert recommendations == {}


def test_collect_meta_rule_recommendations_uses_only_meaningful_values():
    def empty_rule(log):
        return {'preset': None}

    def useful_rule(input_data, input_params, log):
        return {'cv_folds': 3}

    recommendations = collect_meta_rule_recommendations(
        input_data=SimpleNamespace(),
        input_params={'use_meta_rules': True},
        rules=[empty_rule, useful_rule],
        log=_FakeLog(),
    )

    assert recommendations == {'cv_folds': 3}
    assert should_apply_meta_rules({'use_meta_rules': True}) is True
    assert should_apply_meta_rules({'use_meta_rules': False}) is False


def test_build_recommendation_bundle_merges_data_and_param_recommendations():
    def meta_rule(input_data, input_params, log):
        return {'preset': 'fast_train'}

    input_data = SimpleNamespace(
        data_type=DataTypesEnum.table,
        features=np.array([
            ['a', 'x', 1],
            ['b', 'y', 2],
            ['c', 'z', 3],
        ], dtype=object),
    )

    bundle = build_recommendation_bundle(
        input_data=input_data,
        input_params={'use_meta_rules': True},
        safe_mode=True,
        limits=RecommendationLimits(max_size=4, max_cat_cardinality=5),
        categorical_detector=_fake_categorical_detector,
        meta_rules=[meta_rule],
        log=_FakeLog(),
    )

    assert bundle.data == {'cut': {'border': 1}, 'label_encoded': {}}
    assert bundle.params == {'preset': 'fast_train', 'label_encoded': {}}
    assert merge_parameter_recommendations({'label_encoded': {}}, {'cv_folds': 3}) == {
        'cv_folds': 3,
        'label_encoded': {},
    }
