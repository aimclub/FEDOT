from typing import List, Optional

from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import ParentOperator

ERROR_PREFIX = 'Invalid graph configuration:'


class Individual:
    def __init__(self, graph: 'OptGraph', fitness: List[float] = None,
                 parent_operators: List[ParentOperator] = None, computation_time: Optional[int] = None):
        self.parent_operators = parent_operators if parent_operators is not None else []
        self.fitness = fitness
        self.computation_time = computation_time
        self.graph = graph

    def __eq__(self, other):
        return self.graph == other.graph
