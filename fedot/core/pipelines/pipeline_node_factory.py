from random import choice
from typing import Optional

from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptNode
from fedot.core.optimisers.opt_node_factory import OptNodeFactory
from fedot.core.utils import DEFAULT_PARAMS_STUB


class PipelineOptNodeFactory(OptNodeFactory):
    def __init__(self, requirements: PipelineComposerRequirements,
                 advisor: Optional[PipelineChangeAdvisor] = None):
        self.requirements = requirements
        self.advisor = advisor or PipelineChangeAdvisor()

    def exchange_node(self,
                      node: OptNode) -> Optional[OptNode]:
        candidates = self.requirements.secondary if node.nodes_from else self.requirements.primary
        candidates = self.advisor.propose_change(current_operation_id=str(node.content['name']),
                                                 possible_operations=candidates)
        return self._return_node(candidates)

    def get_parent_node(self,
                        node: OptNode,
                        primary: bool) -> Optional[OptNode]:
        parent_operations_ids = None
        possible_operations = self.requirements.primary
        if not primary:
            parent_operations_ids = [str(n.content['name']) for n in node.nodes_from]
            possible_operations = self.requirements.secondary
        candidates = self.advisor.propose_parent(current_operation_id=str(node.content['name']),
                                                 parent_operations_ids=parent_operations_ids,
                                                 possible_operations=possible_operations)
        return self._return_node(candidates)

    def get_node(self,
                 primary: bool) -> Optional[OptNode]:
        candidates = self.requirements.primary if primary else self.requirements.secondary
        return self._return_node(candidates)

    @staticmethod
    def _return_node(candidates) -> Optional[OptNode]:
        if not candidates:
            return None
        return OptNode(content={'name': choice(candidates)})

