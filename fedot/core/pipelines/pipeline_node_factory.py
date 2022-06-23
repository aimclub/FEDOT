from random import choice

from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.optimisers.graph import OptNode
from fedot.core.optimisers.opt_node_factory import NodeFactory
from fedot.core.utils import DEFAULT_PARAMS_STUB


class PipelineOptNodeFactory(NodeFactory):
    def change_node(self,
                    node: OptNode,
                    requirements,
                    advisor: PipelineChangeAdvisor = None):
        candidates = requirements.secondary if node.nodes_from else requirements.primary
        if advisor:
            candidates = advisor.propose_change(current_operation_id=str(node.content['name']),
                                                possible_operations=candidates)
        return self._return_node(candidates)

    def get_separate_parent_node(self,
                                 node: OptNode,
                                 requirements,
                                 advisor: PipelineChangeAdvisor):
        return self._get_parent_node(node, requirements, advisor, separate=True)

    def get_intermediate_parent_node(self,
                                     node: OptNode,
                                     requirements,
                                     advisor: PipelineChangeAdvisor):
        return self._get_parent_node(node, requirements, advisor, separate=False)

    def get_child_node(self,
                       requirements):
        return self._get_node(requirements, primary=False)

    def get_primary_node(self,
                         requirements):
        return self._get_node(requirements, primary=True)

    def _get_parent_node(self,
                         node: OptNode,
                         requirements,
                         advisor: PipelineChangeAdvisor,
                         separate: bool):
        parent_operations_ids = None
        possible_operations = requirements.primary
        if not separate:
            parent_operations_ids = [str(n.content['name']) for n in node.nodes_from]
            possible_operations = requirements.secondary
        candidates = advisor.propose_parent(current_operation_id=str(node.content['name']),
                                            parent_operations_ids=parent_operations_ids,
                                            possible_operations=possible_operations)
        return self._return_node(candidates)

    def _get_node(self,
                  requirements,
                  primary: bool):
        candidates = requirements.primary if primary else requirements.secondary
        return self._return_node(candidates)

    @staticmethod
    def _return_node(candidates):
        if not candidates:
            return None
        return OptNode(content={'name': choice(candidates),
                                'params': DEFAULT_PARAMS_STUB})
