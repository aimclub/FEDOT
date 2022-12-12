from typing import Optional, List, Any
from fedot.core.composer.advisor import DefaultChangeAdvisor, RemoveType
from examples.composite_node import CompositeNode
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.optimisers.opt_node_factory import OptNodeFactory
from fedot.core.repository.graph_operation_reposiroty import GraphOperationRepository
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.dataset_types import DataTypesEnum
from random import choice
from copy import copy

class BnChangeAdvisor(DefaultChangeAdvisor):  
    """
    Class for advising of composite BN changes during evolution
    """

    def propose_change(self, node: CompositeNode, possible_operations: List[str]) -> List[str]:
        """
        Proposes promising candidates for node's parent_model replacement
        :param node: node to propose changes for
        :param possible_operations: list of candidates for replace
        :return: list of candidates with str operations
        """
        parent_types = set([str(n.content['type']) for n in node.nodes_from])
        if not parent_types:
            return node.content['parent_model']
        else:
            tag = 'mix' if all([tp in parent_types for tp in ['disc','cont']]) else 'disc' if ('cont' not in parent_types) else 'cont'
            candidates = get_operations_for_task(self.task, mode='model', tags = [tag])
            candidates = set.intersection(set(candidates), set(possible_operations))
        return list(candidates)

class BnNodeFactory(OptNodeFactory):
    def __init__(self, requirements: Optional[Any] = None,
                 advisor: Optional[BnChangeAdvisor] = None,
                 graph_model_repository: Optional[GraphOperationRepository] = None):
        self.requirements = requirements
        self.advisor = advisor or BnChangeAdvisor()
        self.graph_model_repository = graph_model_repository or self._init_default_graph_model_repo()
    def exchange_node(self, node: CompositeNode = None):
        if not self.graph_model_repository:
            model_repository = OperationTypesRepository
            if not node.nodes_from:
                return node
            else:
                
                candidates = model_repository.suitable_operation(data_type = DataTypesEnum.table)
                models = self.advisor.propose_change(node=node,
                                                            possible_operations=candidates)
                return models#self._return_node(node, candidates)
        else:
            raise NotImplementedError()
    @staticmethod
    def _return_node(node, candidates) -> Optional[CompositeNode]:
        if not candidates:
            return node
        new_content = copy(node.content)
        new_content['parent_model'] = choice(candidates) 
        return CompositeNode(content=new_content)
