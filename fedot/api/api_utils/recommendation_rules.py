from dataclasses import dataclass
from functools import partial
from inspect import signature
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np

from fedot.core.data.input_data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum

MetaRule = Callable[..., Dict[str, Any]]


@dataclass(frozen=True)
class RecommendationLimits:
    max_size: int
    max_cat_cardinality: int


@dataclass(frozen=True)
class RecommendationBundle:
    data: Dict[str, Dict[str, Any]]
    params: Dict[str, Any]


def supports_data_recommendations(input_data: InputData) -> bool:
    return input_data.data_type in (DataTypesEnum.table, DataTypesEnum.text)


def estimate_size_cut_border(input_data: InputData, max_size: int) -> Optional[int]:
    if input_data.data_type != DataTypesEnum.table:
        return None

    rows, columns = input_data.features.shape[0], input_data.features.shape[1]
    if rows * columns <= max_size:
        return None
    return max_size // columns


def estimate_categorical_cardinality(input_data: InputData,
                                     categorical_detector: Callable[[Any], tuple[Sequence[int], Sequence[int]]]) -> int:
    categorical_ids, _ = categorical_detector(input_data.features)
    if not categorical_ids:
        return 0
    return sum(len(np.unique(feature)) for feature in input_data.features[:, categorical_ids].astype(str))


def should_use_label_encoding(input_data: InputData,
                              max_cat_cardinality: int,
                              categorical_detector: Callable[[Any], tuple[Sequence[int], Sequence[int]]]) -> bool:
    return estimate_categorical_cardinality(input_data, categorical_detector) > max_cat_cardinality


def build_safe_data_recommendations(input_data: InputData,
                                    safe_mode: bool,
                                    limits: RecommendationLimits,
                                    categorical_detector: Callable[[Any],
                                                                   tuple[Sequence[int],
                                                                         Sequence[int]]]) -> Dict[str,
                                                                                                  Dict[str,
                                                                                                       Any]]:
    if not safe_mode or not supports_data_recommendations(input_data):
        return {}

    recommendations: Dict[str, Dict[str, Any]] = {}
    border = estimate_size_cut_border(input_data, limits.max_size)
    if border is not None:
        recommendations['cut'] = {'border': border}

    if should_use_label_encoding(input_data, limits.max_cat_cardinality, categorical_detector):
        recommendations['label_encoded'] = {}

    return recommendations


def should_apply_meta_rules(input_params: Optional[Dict[str, Any]]) -> bool:
    return bool(input_params and input_params.get('use_meta_rules'))


def evaluate_meta_rule(rule: MetaRule,
                       input_data: InputData,
                       input_params: Dict[str, Any],
                       log) -> Dict[str, Any]:
    bound_rule = rule
    if 'input_params' in signature(bound_rule).parameters:
        bound_rule = partial(bound_rule, input_params=input_params)
    if 'input_data' in signature(bound_rule).parameters:
        bound_rule = partial(bound_rule, input_data=input_data)
    return bound_rule(log=log)


def collect_meta_rule_recommendations(input_data: InputData,
                                      input_params: Dict[str, Any],
                                      rules: Iterable[MetaRule],
                                      log) -> Dict[str, Any]:
    recommendations: Dict[str, Any] = {}
    if not should_apply_meta_rules(input_params):
        return recommendations

    for rule in rules:
        current_recommendation = evaluate_meta_rule(rule, input_data, input_params, log)
        if any(value is not None and value is not False for value in current_recommendation.values()):
            recommendations.update(current_recommendation)
    return recommendations


def merge_parameter_recommendations(data_recommendations: Dict[str, Dict[str, Any]],
                                    meta_recommendations: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(meta_recommendations)
    if 'label_encoded' in data_recommendations:
        merged['label_encoded'] = data_recommendations['label_encoded']
    return merged


def build_recommendation_bundle(input_data: InputData,
                                input_params: Dict[str, Any],
                                safe_mode: bool,
                                limits: RecommendationLimits,
                                categorical_detector: Callable[[Any], tuple[Sequence[int], Sequence[int]]],
                                meta_rules: Iterable[MetaRule],
                                log) -> RecommendationBundle:
    data_recommendations = build_safe_data_recommendations(
        input_data=input_data,
        safe_mode=safe_mode,
        limits=limits,
        categorical_detector=categorical_detector,
    )
    meta_recommendations = collect_meta_rule_recommendations(
        input_data=input_data,
        input_params=input_params,
        rules=meta_rules,
        log=log,
    )
    params_recommendations = merge_parameter_recommendations(data_recommendations, meta_recommendations)
    return RecommendationBundle(data=data_recommendations, params=params_recommendations)
