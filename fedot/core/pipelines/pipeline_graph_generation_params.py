from typing import Optional, Sequence

from fedot.core.dag.verification_rules import DEFAULT_DAG_RULES
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.dag.graph_verifier import VerifierRuleType
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task


def get_pipeline_generation_params(requirements: Optional[PipelineComposerRequirements] = None,
                                   rules_for_constraint: Sequence[VerifierRuleType] = tuple(DEFAULT_DAG_RULES),
                                   task: Optional[Task] = None) -> GraphGenerationParams:
    if requirements is None and task is not None:
        ops = get_operations_for_task(task)
        requirements = PipelineComposerRequirements(primary=ops, secondary=ops)
    else:
        requirements = requirements or PipelineComposerRequirements()
    advisor = PipelineChangeAdvisor(task)
    node_factory = PipelineOptNodeFactory(requirements, advisor) if requirements else None
    graph_generation_params = GraphGenerationParams(adapter=PipelineAdapter(),
                                                    rules_for_constraint=rules_for_constraint,
                                                    advisor=advisor,
                                                    node_factory=node_factory)
    return graph_generation_params
