from typing import List, Union, Optional, Tuple

from fedot.core.pipelines.node import Node, PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


OpT = Union[str, 'Operation']


class PipelineBuilder:
    """ Builder for incremental construction of directed acyclic Pipelines.
    Semantics:
    - Forward-only & addition-only (can't prepend or delete nodes).
    - Doesn't throw, doesn't fail: methods always have a way to interpret input given current graph state.
    - Is not responsible for the validity of resulting Pipeline (e.g. correct order, valid operations).
    """

    def __init__(self, *initial_nodes: Optional[Node]):
        """ Create builder with prebuilt nodes as origins of the branches. """
        self.heads: List[Node] = list(filter(None, initial_nodes))

    @property
    def _iend(self) -> int:
        return len(self.heads)

    def add_node(self, operation_type: Optional[OpT], branch_idx: int = 0):
        """ Add single node to pipeline branch of specified index.
        If there are no heads => adds single PrimaryNode.
        If there is single head => adds single SecondaryNode using head as input.
        If there are several heads => adds single SecondaryNode using as input the head indexed by head_idx.
        If input is None => do nothing.
        If branch_idx is out of bounds => appends new PrimaryNode.

        :param operation_type: new operation, possibly None
        :param branch_idx: index of the head to use as input for the new node
        :return: self
        """
        if not operation_type:
            return self
        if branch_idx < len(self.heads):
            input_node = self.heads[branch_idx]
            self.heads[branch_idx] = SecondaryNode(operation_type, nodes_from=[input_node])
        else:
            self.heads.append(PrimaryNode(operation_type))
        return self

    def add_sequence(self, *operation_type: OpT, branch_idx: int = 0):
        """ Same as .node() but for many operations at once. """
        for operation in operation_type:
            self.add_node(operation, branch_idx)
        return self

    def grow_branches(self, *operation_type: Optional[OpT]):
        """ Add single node to each branch.

        Argument position means index of the branch to grow.
        None operation means don't grow that branch.
        If there are no nodes => creates new branches.
        If number of input nodes is bigger than number of branches => extra operations create new branches.

        :param operation_type: operations for adding to each branch, maybe None.
        :return: self
        """
        for i, operation in enumerate(operation_type):
            self.add_node(operation, i)
        return self

    def add_branch(self, *operation_type: Optional[OpT], branch_idx: int = 0):
        """ Create branches at the tip of branch with branch_idx.

        None operations are filtered out.
        Number of new branches equals to number of provided operations.
        If there are no heads => will add several primary nodes.
        If there is single head => add several SecondaryNodes using head as the previous.
        If there are several heads => branch head indexed by head_idx.
        If branch_idx is out of bounds => adds PrimaryNodes as new heads at the end.
        If no not-None operations are provided, nothing is changed.

        :param operation_type: operations for new nodes
        :param branch_idx: index of the branch for branching its tip
        :return: self
        """
        operations = list(filter(None, operation_type))
        if not operations:
            return self
        if branch_idx < len(self.heads):
            input_node = self.heads.pop(branch_idx)
            for operation in operations:
                self.heads.insert(branch_idx, SecondaryNode(operation, nodes_from=[input_node]))
        else:
            for operation in operations:
                self.add_node(operation, self._iend)
        return self

    def join_branches(self, operation_type: Optional[OpT]):
        """ Joins all current branches with provided operation as ensemble node.

        If there are no branches => does nothing.
        If there is single branch => adds single SecondaryNode using it as input.
        If there are several branches => adds single SecondaryNode using all heads as inputs.

        :param operation_type: operation to use for joined node
        :return: self
        """
        if self.heads and operation_type:
            new_head = SecondaryNode(operation_type, nodes_from=self.heads)
            self.heads = [new_head]
        return self

    def reset(self):
        """ Reset builder state. """
        self.heads = []

    def to_nodes(self) -> List[Node]:
        """
        Return list of final nodes and reset internal state.
        :return: list of final nodes, possibly empty.
        """
        result = self.heads
        self.reset()
        return result

    def to_pipeline(self) -> Optional[Pipeline]:
        """
        Builds new Pipeline from current tips of branches and resets its state.
        :return: Pipeline if there exist nodes, None if there're no nodes.
        """
        if self.heads:
            built = Pipeline(self.heads)
            self.reset()
            return built
        return None
