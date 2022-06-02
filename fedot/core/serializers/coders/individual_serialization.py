from typing import Any, Dict, Type, TYPE_CHECKING

from . import any_from_json
from fedot.core.optimisers.fitness import SingleObjFitness

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.individual import Individual


def individual_from_json(cls: Type['Individual'], json_obj: Dict[str, Any]) -> 'Individual':
    deserialized = any_from_json(cls, json_obj)
    if isinstance(deserialized.fitness, float):
        deserialized.fitness = SingleObjFitness(deserialized.fitness)
    return deserialized
