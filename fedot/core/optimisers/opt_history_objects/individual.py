from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import uuid4

from fedot.core.dag.graph import Graph
from fedot.core.log import default_log
from fedot.core.optimisers.fitness.fitness import Fitness, null_fitness
from fedot.core.optimisers.graph import OptGraph
from fedot.core.serializers.serializer import default_load, default_save

if TYPE_CHECKING:
    from fedot.core.optimisers.opt_history_objects.parent_operator import ParentOperator


INDIVIDUAL_COPY_RESTRICTION_MESSAGE = ('`Individual` instance was copied.\n'
                                       'Normally, you don\'t want to do that to keep uid-individual uniqueness.\n'
                                       'If this happened during the optimization process, this misusage '
                                       'should be fixed.')


@dataclass(frozen=True)
class Individual:
    graph: Graph
    parent_operator: Optional[ParentOperator] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)
    native_generation: Optional[int] = None
    fitness: Fitness = field(default_factory=null_fitness)
    uid: str = field(default_factory=lambda: str(uuid4()))

    def set_native_generation(self, native_generation):
        if self.native_generation is None:
            super().__setattr__('native_generation', native_generation)

    def _set_fitness_and_graph(self, fitness: Fitness, updated_graph: Optional[OptGraph] = None):
        super().__setattr__('fitness', fitness)
        if updated_graph is not None:
            super().__setattr__('graph', updated_graph)

    def set_evaluation_result(self, eval_result: Union[GraphEvalResult, Fitness],
                              updated_graph: Optional[OptGraph] = None):
        if self.fitness.valid:
            raise ValueError('The individual has valid fitness and can not be evaluated again.')

        if isinstance(eval_result, Fitness):
            self._set_fitness_and_graph(eval_result, updated_graph)
            return

        self._set_fitness_and_graph(eval_result.fitness, eval_result.graph)
        self.metadata.update(eval_result.metadata)

    @property
    def has_native_generation(self) -> bool:
        return self.native_generation is not None

    @property
    def parents(self) -> List[Individual]:
        if not self.parent_operator:
            return []
        return list(self.parent_operator.parent_individuals)

    @property
    def parents_from_prev_generation(self) -> List[Individual]:
        parents_from_prev_generation = []
        next_parents = self.parents
        for _ in range(1_000_000):
            if not next_parents or all(p.has_native_generation for p in next_parents):
                break
            parents = next_parents
            next_parents = []
            for p in parents:
                next_parents += p.parents
        else:  # After the last iteration.
            raise ValueError(f'The individual {self.uid} has invalid inheritance data.')

        parents_from_prev_generation += next_parents
        return parents_from_prev_generation

    @property
    def operators_from_prev_generation(self) -> List[ParentOperator]:
        if not self.parent_operator:
            return []
        parents_from_prev_generation = self.parents_from_prev_generation
        operators = [self.parent_operator]
        next_parents = self.parents
        for _ in range(1_000_000):
            if next_parents == parents_from_prev_generation:
                break
            parents = next_parents
            next_parents = []
            for p in parents:
                next_parents += p.parents
                operators.append(p.parent_operator)
        else:  # After the last iteration.
            raise ValueError(f'The individual {self.uid} has invalid inheritance data.')

        operators.reverse()
        return operators

    def save(self, json_file_path: Union[str, os.PathLike] = None) -> Optional[str]:
        return default_save(obj=self, json_file_path=json_file_path)

    @staticmethod
    def load(json_str_or_file_path: Union[str, os.PathLike] = None) -> Individual:
        return default_load(json_str_or_file_path)

    def __repr__(self):
        return (f'<Individual {self.uid} | fitness: {self.fitness} | native_generation: {self.native_generation} '
                f'| graph: {self.graph}>')

    def __eq__(self, other: Individual):
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


@dataclass
class GraphEvalResult:
    uid_of_individual: str
    fitness: Fitness
    graph: OptGraph  # For the case if evaluation needs to assign some values to the graph.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self):
        return self.fitness.valid
