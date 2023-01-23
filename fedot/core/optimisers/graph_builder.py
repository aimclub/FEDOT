from copy import deepcopy
from typing import Union, Tuple, Optional, List, Dict

from fedot.core.adapter.adapter import DomainStructureType
from fedot.core.dag.graph_node import GraphNode


class GraphBuilder:
    """ Builder for incremental construction of directed acyclic graphs.
    Semantics:
    - Forward-only & addition-only (can't prepend or delete nodes).
    - Doesn't throw, doesn't fail: methods always have a way to interpret input given current graph state.
    - Is not responsible for the validity of resulting graph (e.g. correct order, valid operations).
    - Builds always new graphs (on copies of nodes), preserves its state between builds. State doesn't leak outside.
    """

    OperationType = Union[str, Tuple[str, dict]]

    def __init__(self, graph_adapter: Optional[DomainStructureType] = None, *initial_nodes: Optional[GraphNode]):
        """ Create builder with prebuilt nodes as origins of the branches.
        :param graph_adapter: adapter to adapt built graph to particular graph type.
        """
        self.graph_adapter = graph_adapter
        self.heads: List[GraphNode] = list(filter(None, initial_nodes))

    @property
    def _iend(self) -> int:
        return len(self.heads)

    def reset(self):
        """ Reset builder state. """
        self.heads = []

    def to_nodes(self) -> List[GraphNode]:
        """
        Return list of final nodes and reset internal state.
        :return: list of final nodes, possibly empty.
        """
        return deepcopy(self.heads)

    def add_node(self, operation_type: Optional[str], branch_idx: int = 0, params: Optional[Dict] = None):
        """ Add single node to graph branch of specified index.
        If there are no heads => adds single Node.
        If there is single head => adds single Node using head as input.
        If there are several heads => adds single Node using as input the head indexed by branch_idx.
        If input is None => do nothing.
        If branch_idx is out of bounds => appends new Node.

        :param operation_type: new operation, possibly None
        :param branch_idx: index of the head to use as input for the new node
        :param params: parameters dictionary for the specific operation
        :return: self
        """
        raise NotImplementedError()

    def add_sequence(self, *operation_type: OperationType, branch_idx: int = 0):
        """ Same as .node() but for many operations at once.

        :param operation_type: operations for new nodes, either as an operation name
            or as a tuple of operation name and operation parameters.
        :param branch_idx: index of the branch for branching its tip
        """
        raise NotImplementedError()

    def grow_branches(self, *operation_type: Optional[OperationType]):
        """ Add single node to each branch.

        Argument position means index of the branch to grow.
        None operation means don't grow that branch.
        If there are no nodes => creates new branches.
        If number of input nodes is bigger than number of branches => extra operations create new branches.

        :param operation_type: operations for new nodes, either as an operation name
            or as a tuple of operation name and operation parameters.
        :return: self
        """
        raise NotImplementedError()

    def add_branch(self, *operation_type: Optional[OperationType], branch_idx: int = 0):
        """ Create branches at the tip of branch with branch_idx.

        None operations are filtered out.
        Number of new branches equals to number of provided operations.
        If there are no heads => will add several nodes.
        If there is single head => add several nodes using head as the previous.
        If there are several heads => branch head indexed by branch_idx.
        If branch_idx is out of bounds => adds nodes as new heads at the end.
        If no not-None operations are provided, nothing is changed.

        :param operation_type: operations for new nodes, either as an operation name
            or as a tuple of operation name and operation parameters.
        :param branch_idx: index of the branch for branching its tip
        :return: self
        """
        raise NotImplementedError()

    def add_skip_connection_edge(self, branch_idx_first: int, branch_idx_second: int,
                                 node_idx_in_branch_first: int, node_idx_in_branch_second: int,):
        """ Joins two nodes which are not placed sequential in one branch.
        Can be used only in the very end of graph building but before 'join_branches'.
        Edge is directed from the first node to the second.

        :param branch_idx_first: index of the first branch which to take the first node
        :param branch_idx_second: index of the second branch which to take the second node
        :param node_idx_in_branch_first: index of the node in its branch
        :param node_idx_in_branch_second: index of the node in its branch
        """
        raise NotImplementedError()

    def join_branches(self, operation_type: Optional[str], params: Optional[Dict] = None):
        """ Joins all current branches with provided operation as ensemble node.

        If there are no branches => does nothing.
        If there is single branch => adds single node using it as input.
        If there are several branches => adds single node using all heads as inputs.

        :param operation_type: operation to use for joined node
        :param params: parameters dictionary for the specific operation
        :return: self
        """
        raise NotImplementedError()

    @staticmethod
    def _unpack_operation(operation: Optional[OperationType]) -> Tuple[Optional[str], Optional[Dict]]:
        if isinstance(operation, str) or operation is None:
            return operation, None
        else:
            return operation

    @staticmethod
    def _pack_params(name: str, params: Optional[dict]) -> Optional[dict]:
        return {'name': name, 'params': params} if params else {'name': name}
