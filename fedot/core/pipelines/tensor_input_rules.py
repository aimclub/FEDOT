"""Graph decisions for TensorData inputs; no data arrays or fitting."""
from functools import lru_cache

from golem.core.dag.graph_utils import distance_to_primary_level

from fedot.core.operations.model import Model
from fedot.core.repository.operation_types_repository import get_operation_type_from_id


DECOMPOSE_OPERATIONS = frozenset(('decompose', 'class_decompose'))


def resolve_tensor_target_source(parents, has_target=None) -> int:
    """Preserve legacy main-target priority and shallowest auxiliary fallback.

    A decomposition starts an auxiliary objective. Other nodes inherit the
    objective selected from their parents. The memo only lives for this call,
    so graph mutations cannot leave stale lineage in data or cached state.
    """
    @lru_cache(maxsize=None)
    def has_main_target(node):
        if get_operation_type_from_id(node.operation.operation_type) in DECOMPOSE_OPERATIONS:
            return False
        return not node.nodes_from or any(has_main_target(parent) for parent in node.nodes_from)

    main = [index for index, parent in enumerate(parents) if has_main_target(parent)]
    if main:
        # Ordinary TensorData branches may carry labels in only one modality.
        # Missing labels must not make an auxiliary objective take precedence.
        return next((index for index in main if has_target is None or has_target[index]), main[0])
    return min(range(len(parents)), key=lambda index: distance_to_primary_level(parents[index]))


def resolve_decompose_parents(parents) -> tuple[int, int]:
    """Return prediction/feature parent indices in the supplied execution order."""
    if len(parents) != 2:
        raise ValueError('Decompose requires exactly two parents.')
    models = [index for index, parent in enumerate(parents) if isinstance(parent.operation, Model)]
    if not models:
        raise ValueError('Decompose requires a Model parent.')
    prediction = (models[0] if len(models) == 1 else
                  max(models, key=lambda index: distance_to_primary_level(parents[index])))
    return prediction, 1 - prediction
