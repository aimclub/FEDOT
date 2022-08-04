from typing import Any, Dict, Type

from fedot.core.optimisers.fitness import SingleObjFitness
from fedot.core.optimisers.gp_comp.individual import Individual
from . import any_from_json


def individual_from_json(cls: Type[Individual], json_obj: Dict[str, Any]) -> Individual:
    deserialized = any_from_json(cls, json_obj)
    if isinstance(deserialized.fitness, float):  # legacy histories support
        object.__setattr__(deserialized, 'fitness', SingleObjFitness(deserialized.fitness))
    object.__setattr__(deserialized, 'parent_operators', tuple(deserialized.parent_operators))
    return deserialized
