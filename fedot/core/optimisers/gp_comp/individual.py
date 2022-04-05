from typing import List, Any, Dict
from uuid import uuid4

from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import ParentOperator

ERROR_PREFIX = 'Invalid graph configuration:'


class Individual:
    def __init__(self, graph: 'OptGraph', fitness: List[float] = None, parent_operators: List[ParentOperator] = None,
                 metadata: Dict[str, Any] = None):
        metadata = metadata or {}
        self.uid = str(uuid4())
        self.parent_operators = parent_operators if parent_operators is not None else []
        self.fitness = fitness
        self.graph = graph
        self.metadata: Dict[str, Any] = metadata

    def __eq__(self, other):
        return self.uid == other.uid
