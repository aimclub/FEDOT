from fedot.core.pipelines.pipeline_node_rules import (
    merge_node_parameters,
    normalize_node_parameters,
    should_update_node_parameters,
)
from fedot.core.utils import DEFAULT_PARAMS_STUB, NESTED_PARAMS_LABEL


def test_normalize_node_parameters_handles_default_stub_and_nested_params():
    assert normalize_node_parameters(
        DEFAULT_PARAMS_STUB, DEFAULT_PARAMS_STUB, NESTED_PARAMS_LABEL) == {}
    assert normalize_node_parameters(
        {NESTED_PARAMS_LABEL: {'alpha': 1.0}},
        DEFAULT_PARAMS_STUB,
        NESTED_PARAMS_LABEL,
    ) == {'alpha': 1.0}
    assert normalize_node_parameters(
        {'beta': 2.0}, DEFAULT_PARAMS_STUB, NESTED_PARAMS_LABEL) == {'beta': 2.0}


def test_merge_node_parameters_and_update_rule_are_explicit():
    merged = merge_node_parameters({'alpha': 1.0}, {'beta': 2.0})

    assert merged == {'alpha': 1.0, 'beta': 2.0}
    assert should_update_node_parameters('ridge', ['correct_params']) is True
    assert should_update_node_parameters(
        'atomized_operation', ['correct_params']) is False
    assert should_update_node_parameters('ridge', ['linear']) is False
