from typing import Optional, Sequence

from golem.core.dag.graph_verifier import VerifierRuleType, GraphVerifier
from golem.core.optimisers.optimizer import GraphGenerationParams

from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.pipelines.random_pipeline_factory import RandomPipelineFactory
from fedot.core.pipelines.verification import common_rules
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task


def get_pipeline_generation_params(requirements: Optional[PipelineComposerRequirements] = None,
                                   rules_for_constraint: Sequence[VerifierRuleType] = tuple(common_rules),
                                   task: Optional[Task] = None) -> GraphGenerationParams:
    if requirements is None and task is not None:
        ops = get_operations_for_task(task)
        requirements = PipelineComposerRequirements(primary=ops, secondary=ops)
    else:
        requirements = requirements or PipelineComposerRequirements()
    advisor = PipelineChangeAdvisor(task)
    node_factory = PipelineOptNodeFactory(requirements, advisor) if requirements else None
    adapter = PipelineAdapter()
    verifier = GraphVerifier(rules_for_constraint, adapter)
    random_pipeline_factory = RandomPipelineFactory(verifier, node_factory)
    graph_generation_params = GraphGenerationParams(adapter=adapter,
                                                    rules_for_constraint=rules_for_constraint,
                                                    advisor=advisor,
                                                    node_factory=node_factory,
                                                    random_graph_factory=random_pipeline_factory)
    return graph_generation_params
