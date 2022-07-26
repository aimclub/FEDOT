from copy import deepcopy
from typing import Any, Iterable, List, Optional, Union, Tuple
from uuid import uuid4

from fedot.core.dag.graph_node import GraphNode
from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.dag.node_operator import NodeOperator
from fedot.core.log import default_log
from fedot.core.utilities.data_structures import UniqueList, ensure_wrapped_in_sequence
from fedot.core.utils import DEFAULT_PARAMS_STUB
from fedot.core.visualisation.graph_viz import GraphVisualiser


def node_ops_adaptation(func):
    def _adapt(adapter, node: Any):
        if not isinstance(node, OptNode):
            return adapter.adapt(node)
        return node

    def _decorator(self, *args, **kwargs):
        func_result = func(self, *args, **kwargs)
        self._nodes = [_adapt(self._node_adapter, node) for node in self.nodes]
        return func_result

    return _decorator


class OptNode:
    """
    Class for node definition in optimization graph (OptGraph)

    :param content: content in node (name only or dict with name and params)
    :param nodes_from: parent nodes in directed graph
    """

    def __init__(self, content: Union[str, dict],
                 nodes_from: Optional[List['OptNode']] = None):
        default_dict = {'params': DEFAULT_PARAMS_STUB}

        self.log = default_log(self)

        if isinstance(content, str):
            content = {'name': content}

        for key, value in default_dict.items():
            if key not in content:
                content[key] = value

        self._nodes_from = UniqueList(nodes_from or ())
        self.content = content
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
    """

    def __init__(self, nodes: Optional[Union[OptNode, List[OptNode]]] = None):
        self.log = default_log(self)

        self._nodes = []
        self._operator = GraphOperator(self, self._empty_postproc)

        if nodes:
            for node in ensure_wrapped_in_sequence(nodes):
                self.add_node(node)

    def _empty_postproc(self, nodes=None):
        pass

    @property
    def nodes(self):
        return self._nodes

    @property
    def _node_adapter(self):
        return NodeOperatorAdapter()

    @node_ops_adaptation
    def add_node(self, new_node: OptNode):
        """
        Add new node to the OptGraph

        :param new_node: new OptNode object
        """
        self._operator.add_node(self._node_adapter.restore(new_node))

    @node_ops_adaptation
    def update_node(self, old_node: OptNode, new_node: OptNode):
        """
        Replace old_node with new one.

        :param old_node: OptNode object to replace
        :param new_node: OptNode object to replace
        """

        self._operator.update_node(self._node_adapter.restore(old_node),
                                   self._node_adapter.restore(new_node))

    @node_ops_adaptation
    def delete_node(self, node: OptNode):
        """
        Delete chosen node redirecting all its parents to the child.

        :param node: OptNode object to delete
        """

        self._operator.delete_node(self._node_adapter.restore(node))

    @node_ops_adaptation
    def update_subtree(self, old_subroot: OptNode, new_subroot: OptNode):
        """
        Replace the subtrees with old and new nodes as subroots

        :param old_subroot: OptNode object to replace
        :param new_subroot: OptNode object to replace
        """
        self._operator.update_subtree(self._node_adapter.restore(old_subroot),
                                      self._node_adapter.restore(new_subroot))

    @node_ops_adaptation
    def delete_subtree(self, subroot: OptNode):
        """
        Delete the subtree with node as subroot.

        :param subroot:
        """
        self._operator.delete_subtree(self._node_adapter.restore(subroot))

    def distance_to_root_level(self, node: OptNode) -> int:
        """ Returns distance to root level """
        return self._operator.distance_to_root_level(node=self._node_adapter.restore(node))

    def nodes_from_layer(self, layer_number: int) -> List[Any]:
        """ Returns all nodes from specified layer """
        return self._operator.nodes_from_layer(layer_number=layer_number)

    @node_ops_adaptation
    def node_children(self, node: OptNode) -> List[Optional[OptNode]]:
        """ Returns all node's children """
        return self._operator.node_children(node=self._node_adapter.restore(node))

    @node_ops_adaptation
    def connect_nodes(self, node_parent: OptNode, node_child: OptNode):
        """ Add an edge from node_parent to node_child """
        self._operator.connect_nodes(parent=self._node_adapter.restore(node_parent),
                                     child=self._node_adapter.restore(node_child))

    @node_ops_adaptation
    def disconnect_nodes(self, node_parent: OptNode, node_child: OptNode,
                         is_clean_up_leftovers: bool = True):
        """ Delete an edge from node_parent to node_child """
        self._operator.disconnect_nodes(node_parent=self._node_adapter.restore(node_parent),
                                        node_child=self._node_adapter.restore(node_child),
                                        is_clean_up_leftovers=is_clean_up_leftovers)

    def get_nodes_degrees(self):
        """ Nodes degree as the number of edges the node has:
         k = k(in) + k(out) """
        return self._operator.get_nodes_degrees()

    def get_edges(self):
        """ Returns all available edges in a given graph """
        return self._operator.get_edges()

    def show(self, path: str = None):
        GraphVisualiser().visualise(self, path)

    def __eq__(self, other) -> bool:
        return self._operator.is_graph_equal(other)

    def __str__(self):
        return self._operator.graph_description()

    def __repr__(self):
        return self.__str__()

    @property
    def root_node(self):
        roots = self._operator.root_node()
        return roots

    @property
    def descriptive_id(self):
        return self._operator.descriptive_id

    @property
    def length(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
        return self._operator.graph_depth()

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
