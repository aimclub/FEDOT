from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fedot.core.log import default_log
from fedot.core.optimisers.fitness.fitness import Fitness, null_fitness
from fedot.core.optimisers.graph import OptGraph

INDIVIDUAL_COPY_RESTRICTION_MESSAGE = ('`Individual` instance was copied.\n'
                                       'Normally, you don\'t want to do that to keep uid-individual uniqueness.\n'
                                       'If this happened during the optimization process, this misusage '
                                       'should be fixed.')


@dataclass(frozen=True)
class Individual:
    graph: OptGraph
    parent_operator: Optional[ParentOperator] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)
    native_generation: Optional[int] = None
    fitness: Fitness = field(default_factory=null_fitness)
    uid: str = field(default_factory=lambda: str(uuid4()))

    def set_native_generation(self, native_generation):
        if self.native_generation is None:
            super().__setattr__('native_generation', native_generation)

    def set_evaluation_result(self, fitness: Fitness, updated_graph: Optional[OptGraph] = None):
        if self.fitness.valid:
            raise ValueError('The individual has valid fitness and can not be evaluated again.')
        super().__setattr__('fitness', fitness)
        if updated_graph is not None:
            super().__setattr__('graph', updated_graph)

    @property
    def is_historical(self) -> bool:
        return self.native_generation is not None

    @property
    def parents(self) -> List[Individual]:
        if not self.parent_operator:
            return []
        return list(self.parent_operator.parent_individuals)

    @property
    def historical_parents(self) -> List[Individual]:
        historical_parents = []
        next_parents = self.parents
        while next_parents and not all(p.is_historical for p in next_parents):
            parents = next_parents
            next_parents = []
            for p in parents:
                next_parents += p.parents
                if p.is_historical:
                    historical_parents.append(p)

        historical_parents += next_parents
        return historical_parents

    @property
    def operators_from_prev_generation(self) -> List[ParentOperator]:
        if not self.parent_operator:
            return []
        historical_parents = self.historical_parents
        operators = [self.parent_operator]
        next_parents = self.parents
        while next_parents != historical_parents:
            parents = next_parents
            next_parents = []
            for p in parents:
                next_parents += p.parents
                operators.append(p.parent_operator)

        operators.reverse()
        return operators

    def __repr__(self):
        return (f'<Individual {self.uid} | fitness: {self.fitness} | native_generation: {self.native_generation} '
                f'| graph: {self.graph}>')

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
    type: str
    operators: Tuple[str, ...]
    parent_individuals: Tuple[Individual, ...]
    uid: str = field(default_factory=lambda: str(uuid4()), init=False)

    def __repr__(self):
        return (f'<ParentOperator {self.uid} | type: {self.type} | operators: {self.operators} '
                f'| parent_individuals({len(self.parent_individuals)}): {self.parent_individuals}>')
