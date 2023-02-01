from typing import Union, Tuple, Optional, List, Dict

from fedot.core.adapter.adapter import DomainStructureType, BaseOptimizationAdapter
from fedot.core.dag.linked_graph import LinkedGraph
from fedot.core.optimisers.graph import OptNode, OptGraph
from fedot.core.optimisers.graph_builder import GraphBuilder


class OptGraphBuilder(GraphBuilder):
    """ Builder for incremental construction of directed acyclic OptGraphs.
    Semantics:
    - Forward-only & addition-only (can't prepend or delete nodes).
    - Doesn't throw, doesn't fail: methods always have a way to interpret input given current graph state.
    - Is not responsible for the validity of resulting OptGraph (e.g. correct order, valid operations).
    - Builds always new graphs (on copies of nodes), preserves its state between builds. State doesn't leak outside.
    """

    OperationType = Union[str, Tuple[str, dict]]

    def __init__(self, graph_adapter: Optional[BaseOptimizationAdapter] = None, *initial_nodes: Optional[OptNode]):
        """ Create builder with prebuilt nodes as origins of the branches.
        :param graph_adapter: adapter to adapt built graph to particular graph type.
        """
        super().__init__(graph_adapter, *initial_nodes)
        self.graph_adapter = graph_adapter
        self.heads: List[OptNode] = list(filter(None, initial_nodes))

    def add_node(self, operation_type: Optional[str], branch_idx: int = 0, params: Optional[Dict] = None):
        """ Add single node to graph branch of specified index.
        If there are no heads => adds single OptNode.
        If there is single head => adds single OptNode using head as input.
        If there are several heads => adds single OptNode using as input the head indexed by branch_idx.
        If input is None => do nothing.
        If branch_idx is out of bounds => appends new OptNode.

        :param operation_type: new operation, possibly None
        :param branch_idx: index of the head to use as input for the new node
        :param params: parameters dictionary for the specific operation
        :return: self
        """
        if operation_type is None:
            return self
        params = self._pack_params(operation_type, params)
        if branch_idx < len(self.heads):
            input_node = self.heads[branch_idx]
            self.heads[branch_idx] = OptNode(content=params, nodes_from=[input_node])
        else:
            self.heads.append(OptNode(content=params))
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
        operations = list(filter(None, operation_type))
        if not operations:
            return self
        if branch_idx < len(self.heads):
            input_node = self.heads.pop(branch_idx)
            for i, operation in enumerate(operations):
                operation, params = self._unpack_operation(operation)
                self.heads.insert(branch_idx + i, OptNode(content=self._pack_params(operation, params),
                                                          nodes_from=[input_node]))
        else:
            for operation in operations:
                operation, params = self._unpack_operation(operation)
                self.add_node(operation, self._iend, params)
        return self

    def add_skip_connection_edge(self, branch_idx_first: int, branch_idx_second: int,
                                 node_idx_in_branch_first: int, node_idx_in_branch_second: int):
        """ Joins two nodes which are not placed sequential in one branch.
        Edge is directed from the first node to the second.

        :param branch_idx_first: index of the first branch which to take the first node
        :param branch_idx_second: index of the second branch which to take the second node
        :param node_idx_in_branch_first: index of the node in its branch
        :param node_idx_in_branch_second: index of the node in its branch
        """
        if branch_idx_first >= len(self.heads) or branch_idx_second >= len(self.heads):
            return
        first_node = self._get_node_from_branch_with_idx(branch_idx=branch_idx_first,
                                                         node_idx_in_branch=node_idx_in_branch_first)
        second_node = self._get_node_from_branch_with_idx(branch_idx=branch_idx_second,
                                                          node_idx_in_branch=node_idx_in_branch_second)
        # to avoid cyclic graphs
        if second_node not in first_node.nodes_from and first_node not in second_node.nodes_from:
            second_node.nodes_from.append(first_node)

        return self

    def _get_node_from_branch_with_idx(self, branch_idx: int, node_idx_in_branch: int):
        head_node = self.heads[branch_idx]
        branch_pipeline = OptGraph(head_node)
        return branch_pipeline.nodes[node_idx_in_branch]

    def join_branches(self, operation_type: Optional[str], params: Optional[Dict] = None):
        """ Joins all current branches with provided operation as ensemble node.

        If there are no branches => does nothing.
        If there is single branch => adds single node using it as input.
        If there are several branches => adds single node using all heads as inputs.

        :param operation_type: operation to use for joined node
        :param params: parameters dictionary for the specific operation
        :return: self
        """
        if self.heads and operation_type:
            content = self._pack_params(operation_type, params)
            new_head = OptNode(content=content, nodes_from=self.heads)
            self.heads = [new_head]
        return self

    def build(self) -> Optional[DomainStructureType]:
        """ Adapt resulted graph to required graph class. """
        if not self.to_nodes():
            return None
        result_opt_graph = OptGraph(self.to_nodes())
        if self.graph_adapter:
            return self.graph_adapter.restore(result_opt_graph)
        else:
            return result_opt_graph

    def merge_with(self, following_builder) -> Optional['OptGraphBuilder']:
        return merge_opt_graph_builders(self, following_builder)


def merge_opt_graph_builders(previous: OptGraphBuilder, following: OptGraphBuilder) -> Optional[OptGraphBuilder]:
    """ Merge two builders.

    Merging is defined for cases one-to-many and many-to-one nodes,
    i.e. one final node to many initial nodes and many final nodes to one initial node.
    Merging is undefined for the case of many-to-many nodes and None is returned.
    Merging of the builder with itself is well-defined and leads to duplication of the graph.

    If one of the builders is empty -- the other one is returned, no merging is performed.
    State of the passed builders is preserved as they were, after merging new builder is returned.

    :return: PipelineBuilder if merging is well-defined, None otherwise.
    """

    if not following.heads:
        return previous
    elif not previous.heads:
        return following

    if type(following.graph_adapter) is not type(previous.graph_adapter):
        raise ValueError('Adapters do not match: cannot perform merge')

    lhs_nodes_final = previous.to_nodes()
    rhs_tmp_graph = LinkedGraph(following.to_nodes())
    rhs_nodes_initial = list(filter(lambda node: not node.nodes_from, rhs_tmp_graph.nodes))

    # If merging one-to-one or one-to-many
    if len(lhs_nodes_final) == 1:
        final_node = lhs_nodes_final[0]
        for initial_node in rhs_nodes_initial:
            rhs_tmp_graph.update_node(initial_node,
                                      OptNode(content={'name': initial_node.name}, nodes_from=[final_node]))
    # If merging many-to-one
    elif len(rhs_nodes_initial) == 1:
        initial_node = rhs_nodes_initial[0]
        rhs_tmp_graph.update_node(initial_node,
                                  OptNode(content={'name': initial_node.name}, nodes_from=lhs_nodes_final))
    # Merging is not defined for many-to-many case
    else:
        return None

    # Check that Graph didn't mess up with node types
    if not all(map(lambda n: isinstance(n, OptNode), rhs_tmp_graph.nodes)):
        raise ValueError("Expected Graph only with nodes of type 'OptNode'")

    merged_builder = OptGraphBuilder(following.graph_adapter, *rhs_tmp_graph.root_nodes())
    return merged_builder
