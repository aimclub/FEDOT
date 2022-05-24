from dataclasses import dataclass
from typing import List, Any, Dict, Optional, Union
from uuid import uuid4

from fedot.core.optimisers.fitness.fitness import Fitness, null_fitness
from fedot.core.optimisers.graph import OptGraph

ERROR_PREFIX = 'Invalid graph configuration:'


class Individual:
    def __init__(self, graph: OptGraph,
                 parent_operators: Optional[List['ParentOperator']] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.graph = graph
        self.parent_operators = parent_operators or []
        self.metadata: Dict[str, Any] = metadata or {}
        self.fitness: Fitness = null_fitness()
        self.uid = str(uuid4())

    def __eq__(self, other: 'Individual'):
        return self.uid == other.uid


@dataclass
class ParentOperator:
    operator_name: str
    operator_type: str
    parent_individuals: List[Individual]
    uid: str = None

    def __post_init__(self):
        if not self.uid:
            self.uid = str(uuid4())
