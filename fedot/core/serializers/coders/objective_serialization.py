from typing import Any, Dict, Type, Tuple

from fedot.core.optimisers.objective import Objective
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.serializers.any_serialization import any_from_json


def objective_from_json(cls: Type[Objective], json_obj: Dict[str, Any]) -> Objective:
    # backward compatibility
    metrics = json_obj.get('metrics')
    if metrics and not isinstance(metrics[0], Tuple):
        return any_from_json(MetricsObjective, json_obj)
    return any_from_json(cls, json_obj)
