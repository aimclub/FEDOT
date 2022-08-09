from typing import Any, Optional, Sequence

from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.dag.graph_verifier import VerifierRuleType
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.repository.tasks import Task


def get_pipeline_generation_params(requirements: Any = None,
                                   rules_for_constraint: Sequence[VerifierRuleType] = (),
                                   task: Optional[Task] = None) -> GraphGenerationParams:
    advisor = PipelineChangeAdvisor(task)
    node_factory = PipelineOptNodeFactory(requirements, advisor) if requirements else None
    graph_generation_params = GraphGenerationParams(adapter=PipelineAdapter(),
                                                    rules_for_constraint=rules_for_constraint,
                                                    advisor=advisor,
                                                    node_factory=node_factory)
    return graph_generation_params
