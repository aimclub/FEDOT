from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple, Union
from uuid import uuid4

from fedot.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence


@dataclass(frozen=True)
class ParentOperator:
    type_: str
    operators: Union[Tuple[str, ...], str]
    parent_individuals: Union[Iterable[Individual], Individual] = field()
    uid: str = field(default_factory=lambda: str(uuid4()), init=False)

    def __post_init__(self):
        if isinstance(self.operators, str):
            object.__setattr__(self, 'operators', (self.operators,))
        parent_individuals = ensure_wrapped_in_sequence(self.parent_individuals)
        object.__setattr__(self, 'parent_individuals', tuple(parent_individuals))

    def __repr__(self):
        return (f'<ParentOperator {self.uid} | type: {self.type_} | operators: {self.operators} '
                f'| parent_individuals({len(self.parent_individuals)}): {self.parent_individuals}>')
