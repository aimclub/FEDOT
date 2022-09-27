from typing import Optional, Union, List
from fedot.core.optimisers.graph import OptGraph, OptNode



class CompositeModel(OptGraph):
    def __init__(self, nodes: Optional[Union[OptNode, List[OptNode]]] = None):
        super().__init__(nodes)
        self.unique_pipeline_id = 1