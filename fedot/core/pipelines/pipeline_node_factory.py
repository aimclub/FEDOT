from random import choice
from typing import Optional

from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptNode
from fedot.core.optimisers.opt_node_factory import OptNodeFactory
from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor, check_for_specific_operations
from fedot.core.repository.pipeline_operation_repository import PipelineOperationRepository


class PipelineOptNodeFactory(OptNodeFactory):
    def __init__(self, requirements: PipelineComposerRequirements,
                 advisor: Optional[PipelineChangeAdvisor] = None,
                 graph_model_repository: Optional[PipelineOperationRepository] = None):
        self.requirements = requirements
        self.advisor = advisor or PipelineChangeAdvisor()
        self.graph_model_repository = graph_model_repository or self._init_default_graph_model_repo()

    def _init_default_graph_model_repo(self):
        """ Initialize default graph model repository with operations from composer requirements """
        repo = PipelineOperationRepository(operations_by_keys={'primary': self.requirements.primary,
                                                               'secondary': self.requirements.secondary})
        return repo

    def exchange_node(self,
                      node: OptNode):
        candidates = self.graph_model_repository.get_operations(is_primary=False) \
            if node.nodes_from else self.graph_model_repository.get_operations(is_primary=True)
        candidates = self.advisor.propose_change(node=node,
                                                 possible_operations=candidates)
        candidates = self.filter_specific_candidates(candidates)
        return self._return_node(candidates)

    def get_parent_node(self, node: OptNode, is_primary: bool):
        possible_operations = self.graph_model_repository.get_operations(is_primary=is_primary)
        candidates = self.advisor.propose_parent(node=node,
                                                 possible_operations=possible_operations)
        candidates = self.filter_specific_candidates(candidates)
        return self._return_node(candidates)

    def get_node(self,
                 is_primary: bool):
        candidates = self.graph_model_repository.get_operations(is_primary=is_primary)
        candidates = self.filter_specific_candidates(candidates)
        return self._return_node(candidates)

    @staticmethod
    def _return_node(candidates) -> Optional[OptNode]:
        if not candidates:
            return None
        return OptNode(content={'name': choice(candidates)})

    def filter_specific_candidates(self, candidates: list):
        return list(filter(lambda x: not check_for_specific_operations(x), candidates))
