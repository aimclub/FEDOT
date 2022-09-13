from typing import List, Union

from fedot.core.dag.graph_delegate import GraphDelegate
from fedot.core.dag.graph_node import GraphNode
from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.log import Log, default_log


OptNode = GraphNode


class OptGraph(GraphDelegate):
    """Base class used for optimized structure

    :param nodes: optimization graph nodes object(s)
    """

    def __init__(self, nodes: Union[OptNode, List[OptNode]] = ()):
        super().__init__(nodes)
        self.log = default_log(self)
