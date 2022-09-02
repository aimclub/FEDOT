from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, TypeVar, Generic
from uuid import uuid4

from fedot.core.dag.graph import Graph
from fedot.core.log import default_log
from fedot.core.optimisers.fitness.fitness import Fitness, null_fitness

INDIVIDUAL_COPY_RESTRICTION_MESSAGE = '`Individual` instance was copied.\n' \
                                      'Normally, you don\'t want to do that to keep uid-individual uniqueness.\n' \
                                      'If this happened during the optimization process, this is a misusage ' \
                                      'and should be fixed.'


G = TypeVar('G', bound=Graph)


@dataclass(frozen=True)
class Individual(Generic[G]):
    """Generic immutable container dataclass for Graphs.
    It is used during optimization process for keeping
    fitness and various metadata e.g. computation time."""

    graph: G
    parent_operators: Tuple['ParentOperator', ...] = field(default=())
    metadata: Dict[str, Any] = field(default_factory=dict)
    native_generation: Optional[int] = None
    fitness: Fitness = field(default_factory=null_fitness)
    uid: str = field(default_factory=lambda: str(uuid4()))

    def set_native_generation(self, native_generation):
        if self.native_generation is None:
            super().__setattr__('native_generation', native_generation)

    def set_evaluation_result(self, fitness: Fitness, updated_graph: Optional[G] = None):
        if self.fitness.valid:
            raise ValueError('The individual has valid fitness and can not be evaluated again.')
        super().__setattr__('fitness', fitness)
        if updated_graph is not None:
            super().__setattr__('graph', updated_graph)

    def __eq__(self, other: 'Individual'):
        return self.uid == other.uid

    def __copy__(self):
        default_log(self).warning(INDIVIDUAL_COPY_RESTRICTION_MESSAGE)
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        default_log(self).warning(INDIVIDUAL_COPY_RESTRICTION_MESSAGE)
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            object.__setattr__(result, k, deepcopy(v, memo))
        return result


@dataclass(frozen=True)
class ParentOperator:
    operator_name: str
    operator_type: str
    parent_individuals: Tuple[Individual, ...]
    uid: str = field(default_factory=lambda: str(uuid4()), init=False)
