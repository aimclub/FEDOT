from __future__ import annotations

from collections import UserList
from copy import deepcopy, copy
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Union

from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence

if TYPE_CHECKING:
    from fedot.core.optimisers.opt_history_objects.individual import Individual


class Generation(UserList):
    """
    List of evolution individuals considered as a single generation.
    Allows to provide additional information about the generation.
    Responsible for setting generation-related info to the individuals.
    """

    def __init__(self, iterable: Union[Iterable[Individual], Generation], generation_num: int,
                 label: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.generation_num = generation_num
        self.label: str = label or ''
        self.metadata: Dict[str, Any] = metadata or {}
        super().__init__(iterable)
        self._set_native_generation(self.data)

    def __setitem__(self, index, item: Union[Individual, Iterable[Individual]]):
        super().__setitem__(index, item)
        self._set_native_generation(item)

    def copy(self) -> Generation:
        return copy(self)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.data = copy(self.data)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            object.__setattr__(result, k, deepcopy(v, memo))
        return result

    def _set_native_generation(self, individuals: Union[Individual, Iterable[Individual]]):
        individuals = ensure_wrapped_in_sequence(individuals)
        for individual in individuals:
            individual.set_native_generation(self.generation_num)

    def __repr__(self):
        gen_num = f'Generation {self.generation_num}'
        label = f' ({self.label}): ' if self.label else ': '
        data = super().__repr__()
        return ''.join([gen_num, label, data])
