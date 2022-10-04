from typing import List, Union

from fedot.core.dag.graph_delegate import GraphDelegate
from fedot.core.dag.linked_graph_node import LinkedGraphNode
from fedot.core.log import default_log

OptNode = LinkedGraphNode


class OptGraph(GraphDelegate):
    """Base class used for optimized structure

    Args:
        nodes: optimization graph nodes object(s)
    """

    def __init__(self, nodes: Union[OptNode, List[OptNode]] = ()):
        super().__init__(nodes)
        self.log = default_log(self)
