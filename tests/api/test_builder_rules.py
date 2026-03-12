from fedot.api.builder import DEFAULT_VALUE
from fedot.api.builder_rules import (
    build_fedot_kwargs,
    merge_builder_params,
    normalize_builder_params,
)


def test_normalize_builder_params_skips_default_sentinel():
    update = normalize_builder_params(
        {'timeout': 1, 'preset': DEFAULT_VALUE, 'seed': 42},
        DEFAULT_VALUE,
    )

    assert update.applied_params == {'timeout': 1, 'seed': 42}


def test_merge_builder_params_preserves_existing_values_for_default_updates():
    merged = merge_builder_params(
        current_params={'problem': 'classification', 'timeout': 5},
        new_params={'timeout': DEFAULT_VALUE, 'preset': 'fast_train'},
        default_value=DEFAULT_VALUE,
    )

    assert merged == {'problem': 'classification', 'timeout': 5, 'preset': 'fast_train'}


def test_build_fedot_kwargs_returns_copy():
    api_params = {'problem': 'classification'}
    kwargs = build_fedot_kwargs(api_params)

    assert kwargs == api_params
    assert kwargs is not api_params
