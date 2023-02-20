from typing import Any, Dict, Type, Tuple

from golem.core.optimisers.objective import Objective
from golem.serializers.serializer import Serializer
from golem.serializers.any_serialization import any_from_json
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective


def objective_from_json(cls: Type[Objective], json_obj: Dict[str, Any]) -> Objective:
    # backward compatibility
    metrics = json_obj.get('metrics')
    if metrics and not isinstance(metrics[0], Tuple):
        return any_from_json(MetricsObjective, json_obj)
    return any_from_json(cls, json_obj)


def init_backward_serialize_compat():
    # backward compatibility
    Serializer.register_coders(Objective, from_json=objective_from_json, overwrite=True)
