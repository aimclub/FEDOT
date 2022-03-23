from copy import deepcopy
from typing import Any, List, Optional, Union, Iterable
from uuid import uuid4

from fedot.core.dag.graph_node import GraphNode
from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.dag.node_operator import NodeOperator
from fedot.core.utilities.data_structures import UniqueList
from fedot.core.log import Log, default_log
from fedot.core.utils import DEFAULT_PARAMS_STUB
from fedot.core.visualisation.graph_viz import GraphVisualiser


def node_ops_adaptation(func):
    def _adapt(adapter, node: Any):
        if not isinstance(node, OptNode):
            return adapter.adapt(node)
        return node

    def _decorator(self, *args, **kwargs):
        func_result = func(self, *args, **kwargs)
        self.nodes = [_adapt(self._node_adapter, node) for node in self.nodes]
        return func_result

    return _decorator


class OptNode:
    """
    Class for node definition in optimization graph (OptGraph)

    :param content: content in node (name only or dict with name and params)
    :param nodes_from: parent nodes in directed graph
    """

    def __init__(self, content: Union[str, dict],
                 nodes_from: Optional[List['OptNode']] = None,
                 log: Optional[Log] = None
                 ):
        default_dict = {'params': DEFAULT_PARAMS_STUB}

        self.log = log
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        if isinstance(content, str):
            content = {'name': content}

        self._nodes_from = UniqueList(nodes_from or ())
        self.content = {**content, **default_dict}
        self._operator = NodeOperator(self)
        self.uid = str(uuid4())

    @property
    def nodes_from(self) -> List:
        return self._nodes_from

    @nodes_from.setter
    def nodes_from(self, nodes: Optional[Iterable['OptNode']]):
        self._nodes_from = UniqueList(nodes)

    @property
    def _node_adapter(self):
        return NodeOperatorAdapter()

    def __str__(self):
        return str(self.content['name'])

    def __repr__(self):
        return self.__str__()

    @property
    def descriptive_id(self):
        return self._operator.descriptive_id()

    def ordered_subnodes_hierarchy(self, visited=None) -> List['OptNode']:
        nodes = self._operator.ordered_subnodes_hierarchy(visited)
        return [self._node_adapter.adapt(node) for node in nodes]

    @property
    def distance_to_primary_level(self):
        return self._operator.distance_to_primary_level()


class OptGraph:
    """
    Base class used for optimized structure

    :param nodes: OptNode object(s)
    :param log: Log object to record messages
    """

    def __init__(self, nodes: Optional[Union[OptNode, List[OptNode]]] = None,
                 log: Log = None):
        self.log = log
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        self.nodes = []
        self.operator = GraphOperator(self, self._empty_postproc)

        if nodes:
            if isinstance(nodes, list):
                for node in nodes:
                    self.add_node(node)
            else:
                self.add_node(nodes)

    def _empty_postproc(self, nodes=None):
        pass

    @property
    def _node_adapter(self):
        return NodeOperatorAdapter()

    @node_ops_adaptation
    def add_node(self, new_node: OptNode):
        """
        Add new node to the OptGraph

        :param new_node: new OptNode object
        """
        self.operator.add_node(self._node_adapter.restore(new_node))

    @node_ops_adaptation
    def update_node(self, old_node: OptNode, new_node: OptNode):
        """
        Replace old_node with new one.

        :param old_node: OptNode object to replace
        :param new_node: OptNode object to replace
        """

        self.operator.update_node(self._node_adapter.restore(old_node),
                                  self._node_adapter.restore(new_node))

    @node_ops_adaptation
    def update_subtree(self, old_subroot: OptNode, new_subroot: OptNode):
        """
        Replace the subtrees with old and new nodes as subroots

        :param old_subroot: OptNode object to replace
        :param new_subroot: OptNode object to replace
        """
        self.operator.update_subtree(self._node_adapter.restore(old_subroot),
                                     self._node_adapter.restore(new_subroot))

    @node_ops_adaptation
    def delete_node(self, node: OptNode):
        """
        Delete chosen node redirecting all its parents to the child.

        :param node: OptNode object to delete
        """

        self.operator.delete_node(self._node_adapter.restore(node))

    @node_ops_adaptation
    def delete_subtree(self, subroot: OptNode):
        """
        Delete the subtree with node as subroot.

        :param subroot:
        """
        self.operator.delete_subtree(self._node_adapter.restore(subroot))

    def show(self, path: str = None):
        GraphVisualiser().visualise(self, path)

    def __eq__(self, other) -> bool:
        return self.operator.is_graph_equal(other)

    def __str__(self):
        return self.operator.graph_description()

    def __repr__(self):
        return self.__str__()

    @property
    def root_node(self):
        roots = self.operator.root_node()
        return roots

    @property
    def length(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
        return self.operator.graph_depth()

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo=None):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class NodeOperatorAdapter:
    def adapt(self, adaptee) -> OptNode:
        adaptee.__class__ = OptNode
        return adaptee

    def restore(self, node) -> GraphNode:
        obj = node
        obj.__class__ = GraphNode
        return obj
