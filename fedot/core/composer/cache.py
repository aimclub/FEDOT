import re
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, TypeVar, Union

from fedot.core.composer.cache_db import OperationsCacheDB
from fedot.core.log import Log, SingletonMeta, default_log
from fedot.core.operations.operation import Operation
from fedot.core.pipelines.node import Node
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline

IOperation = TypeVar('IOperation', bound=Operation)


@dataclass
class CachedState:
    operation: IOperation


class OperationsCache(metaclass=SingletonMeta):
    """
    Stores/loads nodes `fitted_operation` field to increase performance of calculations.

    :param log: optional Log object to record messages
    :param db_path: optional str db file path
    """

    def __init__(self, log: Optional[Log] = None, db_path: Optional[str] = None):
        self.log = log or default_log(__name__)
        self._db = OperationsCacheDB(db_path)

    @property
    def effectiveness_ratio(self):
        """
        Returns percent of how many pipelines/nodes were loaded instead of computing
        """
        #  Result order corresponds to the order in self.db._effectiveness_keys
        pipelines_hit, nodes_hit, pipelines_total, nodes_total = self._db.get_effectiveness()

        return {
            'pipelines': round(pipelines_hit / pipelines_total, 3) if pipelines_total else 0.,
            'nodes': round(nodes_hit / nodes_total, 3) if nodes_total else 0.
        }

    def reset(self):
        self._db.reset()

    def save_nodes(self, nodes: Union[Node, List[Node]], fold_id: Optional[int] = None):
        """
        :param nodes: node/nodes for caching
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        try:
            mapped = [
                (_get_structural_id(node, fold_id), CachedState(node.fitted_operation))
                for node in ensure_wrapped_in_sequence(nodes)
                if node.fitted_operation is not None
            ]
            self._db.add_operations(mapped)
        except Exception as ex:
            self.log.info(f'Nodes can not be saved: {ex}. Continue')

    def save_pipeline(self, pipeline: 'Pipeline', fold_id: Optional[int] = None):
        """
        :param pipeline: pipeline for caching
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        self.save_nodes(pipeline.nodes, fold_id)

    def try_load_nodes(self, nodes: Union[Node, List[Node]], fold_id: Optional[int] = None) -> bool:
        """
        :param nodes: nodes which fitted state should be loaded from cache
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        cache_was_used = False
        try:
            nodes_lst = ensure_wrapped_in_sequence(nodes)
            structural_ids = [_get_structural_id(node, fold_id) for node in nodes_lst]
            cached_states = self._db.get_operations(structural_ids)
            for idx, cached_state in enumerate(cached_states):
                if cached_state is not None:
                    nodes_lst[idx].fitted_operation = cached_state.operation
                    cache_was_used = True
                else:
                    nodes_lst[idx].fitted_operation = None
        except Exception as ex:
            self.log.info(f'Cache can not be loaded: {ex}. Continue.')
        finally:
            return cache_was_used

    def try_load_into_pipeline(self, pipeline: 'Pipeline', fold_id: Optional[int] = None) -> bool:
        """
        :param pipeline: pipeline for loading cache into
        :param fold_id: optional part of cache item UID
                            (number of the CV fold)
        """
        return self.try_load_nodes(pipeline.nodes, fold_id)

    def __len__(self):
        return len(self._db)


def _get_structural_id(node: Node, fold_id: Optional[int] = None) -> str:
    structural_id = re.sub(f'[{string.punctuation}]+', '', node.descriptive_id)
    structural_id += f'_{fold_id}' if fold_id is not None else ''
    return structural_id
