from typing import List, Any, Dict, Optional, Union
from uuid import uuid4

from fedot.core.optimisers.fitness.fitness import Fitness, none_fitness
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import ParentOperator

ERROR_PREFIX = 'Invalid graph configuration:'


class Individual:
    def __init__(self, graph: 'OptGraph', fitness: Fitness = None,  # TODO: fitness field is never set in init here
                 parent_operators: Optional[List[ParentOperator]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.uid = str(uuid4())
        self.parent_operators = parent_operators or []
        self.fitness = fitness or none_fitness()
        self.graph = graph
        self.metadata: Dict[str, Any] = metadata or {}

    def __eq__(self, other: 'Individual'):
        return self.uid == other.uid
