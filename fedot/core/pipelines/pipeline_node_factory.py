from random import choice

from fedot.core.composer.advisor import DefaultChangeAdvisor
from fedot.core.optimisers.graph import OptNode
from fedot.core.optimisers.opt_node_factory import NodeFactory
from fedot.core.utils import DEFAULT_PARAMS_STUB


class PipelineNodeFactory(NodeFactory):
    def change_node(self,
                    node: OptNode,
                    requirements):
        candidates = requirements.secondary if node.nodes_from else requirements.primary
        return self.return_node(candidates)

    def change_intermediate_node(self,
                                 node: OptNode,
                                 requirements,
                                 advisor: DefaultChangeAdvisor):
        candidates = requirements.secondary if node.nodes_from else requirements.primary
        candidates = advisor.propose_change(current_operation_id=str(node.content['name']),
                                            possible_operations=candidates)
        return self.return_node(candidates)

    def get_parent_node(self,
                        node: OptNode,
                        requirements,
                        advisor: DefaultChangeAdvisor):
        candidates = advisor.propose_parent(current_operation_id=str(node.content['name']),
                                            parent_operations_ids=None,
                                            possible_operations=requirements.primary)
        return self.return_node(candidates)

    def get_child_node(self,
                       requirements):
        candidates = requirements.secondary
        return self.return_node(candidates)

    def get_intermediate_node(self,
                              node: OptNode,
                              requirements,
                              advisor: DefaultChangeAdvisor):
        candidates = advisor.propose_parent(current_operation_id=str(node.content['name']),
                                            parent_operations_ids=[str(n.content['name']) for n in node.nodes_from],
                                            possible_operations=requirements.secondary)
        return self.return_node(candidates)

    def get_primary_node(self, requirements):
        candidates = requirements.primary
        return self.return_node(candidates)

    @staticmethod
    def return_node(candidates):
        return OptNode(content={'name': choice(candidates),
                                'params': DEFAULT_PARAMS_STUB})

