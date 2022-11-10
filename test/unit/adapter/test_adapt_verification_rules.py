import pytest

from fedot.core.dag.verification_rules import DEFAULT_DAG_RULES
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.verification_rules import *
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

SOME_PIPELINE_RULES = (
    has_correct_operations_for_task,
    has_primary_nodes,
    has_no_conflicts_with_data_flow,
    has_correct_data_connections,
)


def get_valid_pipeline():
    pipeline = PipelineBuilder().add_sequence('logit', 'logit', 'logit').to_pipeline()
    adapter = PipelineAdapter()
    opt_graph = adapter.adapt(pipeline)
    return opt_graph, pipeline, adapter


@pytest.mark.parametrize('rule', DEFAULT_DAG_RULES)
def test_adapt_verification_rules_dag(rule):
    """Test that dag verification rules behave as expected with new adapter.
    They accept any graphs, so the new adapter must see them as native
    and shouldn't change them on the call to adapt."""

    opt_graph, pipeline, adapter = get_valid_pipeline()
    adapted_rule = adapter.adapt_func(rule)

    assert adapted_rule(opt_graph)
    assert id(rule) == id(adapted_rule)


@pytest.mark.parametrize('rule', SOME_PIPELINE_RULES)
def test_adapt_verification_rules_pipeline(rule):
    """Test that pipeline verification rules behave as expected with new adapter."""

    opt_graph, pipeline, adapter = get_valid_pipeline()

    # sanity check
    assert rule(pipeline)

    adapted_rule = adapter.adapt_func(rule)

    # adapted rules can accept both opt graphs and pipelines
    assert adapted_rule(opt_graph)
    assert adapted_rule(pipeline)
