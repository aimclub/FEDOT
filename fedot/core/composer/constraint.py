from copy import deepcopy
from typing import Optional

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.graph import OptGraph
from fedot.core.pipelines.validation import validate


def constraint_function(graph: OptGraph,
                        params: Optional['GraphGenerationParams'] = None):
    try:
        rules = params.rules_for_constraint if params else None
        object_for_validation = params.adapter.restore(deepcopy(graph))
        validate(object_for_validation, rules, params.advisor.task)
        return True
    except ValueError:
        return False
