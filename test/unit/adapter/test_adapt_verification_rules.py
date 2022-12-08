import pytest

from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.verification_rules import *

SOME_PIPELINE_RULES = (
    has_correct_operations_for_task,
    has_primary_nodes,
    has_no_conflicts_with_data_flow,
    has_correct_data_connections,
)


def get_valid_pipeline():
    pipeline = PipelineBuilder().add_sequence('logit', 'logit', 'logit').build()
    adapter = PipelineAdapter()
    opt_graph = adapter.adapt(pipeline)
    return opt_graph, pipeline, adapter


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
