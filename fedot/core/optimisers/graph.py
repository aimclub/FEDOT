from typing import List, Union

from fedot.core.dag.graph_delegate import GraphDelegate
from fedot.core.dag.graph_node import GraphNode
from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.log import Log, default_log
from fedot.core.utilities.data_structures import Copyable

OptNode = GraphNode


class OptGraph(GraphDelegate, Copyable):
    """Base class used for optimized structure

    :param nodes: optimization graph nodes object(s)
    """

    def __init__(self, nodes: Union[OptNode, List[OptNode]] = ()):
        self.log = default_log(self)
        super().__init__(GraphOperator(nodes))
