from dataclasses import dataclass
from typing import Any, Dict, List, Optional
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
        self.pop_num = None
        self.ind_num = None

    @property
    def positional_id(self) -> str:
        """
        Identified for location of individual in history of population-based optimisation
        :return: string representation of population number and number of individual in population
        """
        return f'g{self.pop_num}-i{self.ind_num}'

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
