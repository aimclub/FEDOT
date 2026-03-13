import argparse

import numpy as np
import pandas as pd
import pytest

from examples.benchmark.run_amlb import (
    _build_config_from_args,
    _json_ready,
    _resolve_dataset_specs,
    _sanitize_features_for_fedot,
    parse_ratio_list,
)


def test_parse_ratio_list_sorts_and_deduplicates_values():
    ratios = parse_ratio_list('0.3, 0.1,0.3,0.2')
    assert ratios == (0.1, 0.2, 0.3)


def test_parse_ratio_list_rejects_invalid_range():
    with pytest.raises(ValueError, match='Candidate ratio'):
        parse_ratio_list('0.0,0.2')


def test_resolve_dataset_specs_from_category_profile():
    specs = _resolve_dataset_specs(dataset_names=(), amlb_categories=('small_samples_many_classes',))

    assert len(specs) > 0
    assert specs[0].name.startswith('amlb_')


def test_resolve_dataset_specs_rejects_unknown_dataset_name():
    with pytest.raises(ValueError, match='Unknown AMLB dataset profile'):
        _resolve_dataset_specs(dataset_names=('amlb_unknown_dataset',), amlb_categories=())


def test_json_ready_converts_numpy_scalars_and_arrays():
    payload = {
        'arr': np.array([1, 2]),
        'int': np.int64(10),
        'float': np.float64(1.5),
        'bool': np.bool_(True),
    }

    converted = _json_ready(payload)

    assert converted['arr'] == [1, 2]
    assert converted['int'] == 10
    assert converted['float'] == 1.5
    assert converted['bool'] is True


def test_build_config_from_args_uses_15_min_default_and_sampling_values():
    args = argparse.Namespace(
        datasets=[],
        amlb_categories=['amlb_top20_mix'],
        timeout_minutes=15.0,
        seed=7,
        n_jobs=-1,
        preset='best_quality',
        disable_tuning=False,
        max_rows=12345,
        output_root='examples/benchmark/results',
        disable_baseline=False,
        disable_sampling=False,
        sampling_strategy='random',
        sampling_strategy_params_json='{"rank": 16}',
        candidate_ratios='0.5,0.2',
        delta_threshold=0.02,
        cap_max_timeout_share=0.25,
    )

    config = _build_config_from_args(args)

    assert config.timeout_minutes_per_dataset == pytest.approx(15.0)
    assert config.sampling_config['strategy_kind'] == 'subset'
    assert config.sampling_config['random_state'] == 7
    assert config.sampling_config['strategy'] == 'random'
    assert config.sampling_config['candidate_ratios'] == [0.2, 0.5]
    assert config.sampling_config['strategy_params'] == {'rank': 16}


def test_sanitize_features_replaces_pandas_na_values_for_fedot_compatibility():
    frame = pd.DataFrame({
        'num_feature': [1.0, pd.NA, np.nan, 4.0],
        'cat_feature': ['a', pd.NA, 'b', None],
    })

    sanitized = _sanitize_features_for_fedot(
        features=frame,
        numeric_columns=['num_feature'],
        categorical_columns=['cat_feature'],
    )

    assert int(sanitized['num_feature'].isna().sum()) == 0
    assert int(sanitized['cat_feature'].isna().sum()) == 0
    assert np.issubdtype(sanitized['cat_feature'].dtype, np.integer)

    unique_values = np.unique(sanitized['cat_feature'].to_numpy())
    assert len(unique_values) >= 2
    assert all(dtype.kind in {'i', 'u', 'f'} for dtype in sanitized.dtypes)
