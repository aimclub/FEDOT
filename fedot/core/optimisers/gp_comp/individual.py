from typing import List

from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import ParentOperator
from fedot.shared import BasicSerializer

ERROR_PREFIX = 'Invalid graph configuration:'


class Individual(BasicSerializer):
    def __init__(self, graph: 'OptGraph', fitness: List[float] = None,
                 parent_operators: List[ParentOperator] = None):
        self.parent_operators = parent_operators if parent_operators is not None else []
        self.fitness = fitness
        self.graph = graph

    def __eq__(self, other):
        return self.graph == other.graph
