from random import choice
from typing import Optional

from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptNode
from fedot.core.optimisers.opt_node_factory import OptNodeFactory


class PipelineOptNodeFactory(OptNodeFactory):
    def __init__(self, requirements: PipelineComposerRequirements,
                 advisor: Optional[PipelineChangeAdvisor] = None):
        self.requirements = requirements
        self.advisor = advisor or PipelineChangeAdvisor()

    def exchange_node(self,
                      node: OptNode) -> Optional[OptNode]:
        candidates = self.requirements.secondary if node.nodes_from else self.requirements.primary
        candidates = self.advisor.propose_change(node=node,
                                                 possible_operations=candidates)
        return self._return_node(candidates)

    def get_parent_node(self,
                        node: OptNode,
                        is_primary: bool) -> Optional[OptNode]:
        possible_operations = self.requirements.primary
        if not is_primary:
            possible_operations = self.requirements.secondary
        candidates = self.advisor.propose_parent(node=node,
                                                 possible_operations=possible_operations)
        return self._return_node(candidates)

    def get_node(self,
                 is_primary: bool):
        candidates = self.requirements.primary if is_primary else self.requirements.secondary
        return self._return_node(candidates)

    @staticmethod
    def _return_node(candidates) -> Optional[OptNode]:
        if not candidates:
            return None
        return OptNode(content={'name': choice(candidates)})
