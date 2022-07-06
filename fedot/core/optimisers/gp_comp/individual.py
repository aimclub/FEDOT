from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fedot.core.optimisers.fitness.fitness import Fitness, null_fitness
from fedot.core.optimisers.graph import OptGraph

ERROR_PREFIX = 'Invalid graph configuration:'


@dataclass(frozen=True)
class Individual:
    graph: OptGraph
    parent_operators: List['ParentOperator'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    native_generation: Optional[int] = None
    fitness: Fitness = field(default_factory=null_fitness, init=False)
    uid: str = field(default_factory=lambda: str(uuid4()), init=False)

    def __eq__(self, other: 'Individual'):
        return self.uid == other.uid

    def set_native_generation(self, native_generation):
        if self.native_generation is None:
            super().__setattr__('native_generation', native_generation)

    def set_fitness(self, fitness: Fitness):
        if self.fitness.valid:
            raise ValueError
        super().__setattr__('fitness', fitness)

    def set_fitness_and_graph(self, fitness: Fitness, graph: OptGraph):
        if self.fitness.valid:
            raise ValueError
        super().__setattr__('fitness', fitness)
        super().__setattr__('graph', graph)

    def set_uid(self, uid: str):
        super().__setattr__('uid', uid)

    def __copy__(self):
        raise ValueError

    def __deepcopy__(self, memodict=None):
        raise ValueError


@dataclass
class ParentOperator:
    operator_name: str
    operator_type: str
    parent_individuals: List[Individual]
    uid: str = None

    def __post_init__(self):
        if not self.uid:
            self.uid = str(uuid4())
