"""Attempt-local prediction writes, published only after successful evaluation."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FittedOperationSnapshot:
    descriptive_id: str
    fitted_operation: Any


def snapshot_operations(graph):
    """Hold fitted references until cleanup succeeds; node.unfit only detaches them."""
    return [FittedOperationSnapshot(node.descriptive_id, node.fitted_operation) for node in graph.nodes]


class PredictionCacheSession:
    """Duck-compatible with PredictionsCache; owns only its pending writes.

    The shared cache remains owned by the composer. A failed attempt drops all
    local references instead of deleting other workers' durable cache entries.
    """

    def __init__(self, cache):
        self.cache = cache
        self.pending = {}

    def save_node_prediction(self, descriptive_id, output_mode, fold_id, outputData, is_fit=False):
        if self.cache is None:
            raise RuntimeError('prediction cache attempt is closed')
        self.pending[(descriptive_id, output_mode, fold_id, is_fit)] = deepcopy(outputData)

    def load_node_prediction(self, descriptive_id, output_mode, fold_id, is_fit=False):
        if self.cache is None:
            raise RuntimeError('prediction cache attempt is closed')
        key = (descriptive_id, output_mode, fold_id, is_fit)
        if key in self.pending:
            return deepcopy(self.pending[key])
        return self.cache.load_node_prediction(descriptive_id, output_mode, fold_id, is_fit)

    def commit(self):
        for (node, mode, fold, is_fit), data in self.pending.items():
            self.cache.save_node_prediction(node, mode, fold, data, is_fit)

    def close(self):
        self.pending.clear()
        self.cache = None
