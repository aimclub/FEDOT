from typing import Any, Dict, Type

from fedot.core.optimisers.fitness.fitness import Fitness
from . import any_from_json


def fitness_from_json(cls: Type[Fitness], dump: Dict[str, Any]) -> Fitness:
    obj = any_from_json(cls, dump)
    obj.values = obj.values  # invoke values.setter properly
    return obj
