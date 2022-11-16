from copy import deepcopy
from typing import List, Union, Optional, Tuple, Dict

from fedot.core.dag.linked_graph import LinkedGraph
from fedot.core.pipelines.node import Node, PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


class PipelineBuilder:
    """ Builder for incremental construction of directed acyclic Pipelines.
    Semantics:
    - Forward-only & addition-only (can't prepend or delete nodes).
    - Doesn't throw, doesn't fail: methods always have a way to interpret input given current graph state.
    - Is not responsible for the validity of resulting Pipeline (e.g. correct order, valid operations).
    - Builds always new pipelines (on copies of nodes), preserves its state between builds. State doesn't leak outside.
    """

    OperationType = Union[str, Tuple[str, dict]]

    def __init__(self, *initial_nodes: Optional[Node]):
        """ Create builder with prebuilt nodes as origins of the branches. """
        self.heads: List[Node] = list(filter(None, initial_nodes))

    @property
    def _iend(self) -> int:
        return len(self.heads)

    def add_node(self, operation_type: Optional[str], branch_idx: int = 0, params: Optional[Dict] = None):
        """ Add single node to pipeline branch of specified index.
        If there are no heads => adds single PrimaryNode.
        If there is single head => adds single SecondaryNode using head as input.
        If there are several heads => adds single SecondaryNode using as input the head indexed by head_idx.
        If input is None => do nothing.
        If branch_idx is out of bounds => appends new PrimaryNode.

        :param operation_type: new operation, possibly None
        :param branch_idx: index of the head to use as input for the new node
        :param params: parameters dictionary for the specific operation
        :return: self
        """
        if not operation_type:
            return self
        params = self._pack_params(operation_type, params)
        if branch_idx < len(self.heads):
            input_node = self.heads[branch_idx]
            self.heads[branch_idx] = SecondaryNode(operation_type, nodes_from=[input_node], content=params)
        else:
            self.heads.append(PrimaryNode(operation_type, content=params))
        return self

    def add_sequence(self, *operation_type: OperationType, branch_idx: int = 0):
        """ Same as .node() but for many operations at once.

        :param operation_type: operations for new nodes, either as an operation name
            or as a tuple of operation name and operation parameters.
        :param branch_idx: index of the branch for branching its tip
        """
        for operation in operation_type:
            operation, params = self._unpack_operation(operation)
            self.add_node(operation, branch_idx, params)
        return self

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
        for i, operation in enumerate(operation_type):
            operation, params = self._unpack_operation(operation)
            self.add_node(operation, i, params)
        return self

    def add_branch(self, *operation_type: Optional[OperationType], branch_idx: int = 0):
        """ Create branches at the tip of branch with branch_idx.

        None operations are filtered out.
        Number of new branches equals to number of provided operations.
        If there are no heads => will add several primary nodes.
        If there is single head => add several SecondaryNodes using head as the previous.
        If there are several heads => branch head indexed by head_idx.
        If branch_idx is out of bounds => adds PrimaryNodes as new heads at the end.
        If no not-None operations are provided, nothing is changed.

        :param operation_type: operations for new nodes, either as an operation name
            or as a tuple of operation name and operation parameters.
        :param branch_idx: index of the branch for branching its tip
        :return: self
        """
        operations = list(filter(None, operation_type))
        if not operations:
            return self
        if branch_idx < len(self.heads):
            input_node = self.heads.pop(branch_idx)
            for i, operation in enumerate(operations):
                operation, params = self._unpack_operation(operation)
                self.heads.insert(branch_idx + i, SecondaryNode(operation,
                                                                nodes_from=[input_node],
                                                                content=self._pack_params(operation, params)))
        else:
            for operation in operations:
                operation, params = self._unpack_operation(operation)
                self.add_node(operation, self._iend, params)
        return self

    def join_branches(self, operation_type: Optional[str], params: Optional[Dict] = None):
        """ Joins all current branches with provided operation as ensemble node.

        If there are no branches => does nothing.
        If there is single branch => adds single SecondaryNode using it as input.
        If there are several branches => adds single SecondaryNode using all heads as inputs.

        :param operation_type: operation to use for joined node
        :param params: parameters dictionary for the specific operation
        :return: self
        """
        if self.heads and operation_type:
            content = self._pack_params(operation_type, params)
            new_head = SecondaryNode(operation_type, nodes_from=self.heads, content=content)
            self.heads = [new_head]
        return self

    def reset(self):
        """ Reset builder state. """
        self.heads = []

    def merge_with(self, following_builder) -> Optional['PipelineBuilder']:
        return merge_pipeline_builders(self, following_builder)

    def to_nodes(self) -> List[Node]:
        """
        Return list of final nodes and reset internal state.
        :return: list of final nodes, possibly empty.
        """
        return deepcopy(self.heads)

    def to_pipeline(self) -> Optional[Pipeline]:
        """
        Builds new Pipeline from current tips of branches. Preserves builder state.
        :return: Pipeline if there exist nodes, None if there're no nodes.
        """
        return Pipeline(self.to_nodes()) if self.heads else None

    @staticmethod
    def _unpack_operation(operation: Optional[OperationType]) -> Tuple[Optional[str], Optional[Dict]]:
        if isinstance(operation, str) or operation is None:
            return operation, None
        else:
            return operation

    @staticmethod
    def _pack_params(name: str, params: Optional[dict]) -> Optional[dict]:
        return {'name': name, 'params': params} if params else None


def merge_pipeline_builders(previous: PipelineBuilder, following: PipelineBuilder) -> Optional[PipelineBuilder]:
    """ Merge two pipeline builders.

    Merging is defined for cases one-to-many and many-to-one nodes,
    i.e. one final node to many initial nodes and many final nodes to one initial node.
    Merging is undefined for the case of many-to-many nodes and None is returned.
    Merging of the builder with itself is well-defined and leads to duplication of the pipeline.

    If one of the builders is empty -- the other one is returned, no merging is performed.
    State of the passed builders is preserved as they were, after merging new builder is returned.

    :return: PipelineBuilder if merging is well-defined, None otherwise.
    """

    if not following.heads:
        return previous
    elif not previous.heads:
        return following

    lhs_nodes_final = previous.to_nodes()
    rhs_tmp_graph = LinkedGraph(following.to_nodes())
    rhs_nodes_initial = list(filter(lambda node: not node.nodes_from, rhs_tmp_graph.nodes))

    # If merging one-to-one or one-to-many
    if len(lhs_nodes_final) == 1:
        final_node = lhs_nodes_final[0]
        for initial_node in rhs_nodes_initial:
            rhs_tmp_graph.update_node(initial_node,
                                      SecondaryNode(initial_node.operation.operation_type, nodes_from=[final_node]))
    # If merging many-to-one
    elif len(rhs_nodes_initial) == 1:
        initial_node = rhs_nodes_initial[0]
        rhs_tmp_graph.update_node(initial_node,
                                  SecondaryNode(initial_node.operation.operation_type, nodes_from=lhs_nodes_final))
    # Merging is not defined for many-to-many case
    else:
        return None

    # Check that Graph didn't mess up with node types
    if not all(map(lambda n: isinstance(n, Node), rhs_tmp_graph.nodes)):
        raise ValueError("Expected Graph only with nodes of type 'Node'")

    merged_builder = PipelineBuilder(*rhs_tmp_graph.root_nodes())
    return merged_builder
