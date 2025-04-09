import sqlite3
from typing import List, Optional, TYPE_CHECKING, Union

from golem.utilities.data_structures import ensure_wrapped_in_sequence

from fedot.core.caching.base_cache import BaseCache
from fedot.core.caching.operations_cache_db import OperationsCacheDB
from fedot.core.pipelines.node import PipelineNode

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


class OperationsCache(BaseCache):
    """
    Stores/loads nodes `fitted_operation` field to increase performance of calculations.

    :param cache_dir: path to the place where cache files should be stored.
    """

    def __init__(self, cache_dir: Optional[str] = None, custom_pid=None, use_stats: bool = False):
        super().__init__(OperationsCacheDB(cache_dir, custom_pid, use_stats))

    def save_nodes(self, nodes: Union[PipelineNode, List[PipelineNode]], fold_id: Optional[int] = None):
        """
        :param nodes: node/nodes to be cached
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        try:
            mapped = [
                (_get_structural_id(node, fold_id), node.fitted_operation)
                for node in ensure_wrapped_in_sequence(nodes)
                if node.fitted_operation is not None
            ]
            self._db.add_operations(mapped)
        except Exception as ex:
            unexpected_exc = not (isinstance(ex, sqlite3.DatabaseError) and 'disk is full' in str(ex))
            self.log.warning(f'Nodes can not be saved: {ex}. Continue', exc=ex, raise_if_test=unexpected_exc)

    def save_pipeline(self, pipeline: 'Pipeline', fold_id: Optional[int] = None):
        """
        :param pipeline: pipeline to be cached
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        self.save_nodes(pipeline.nodes, fold_id)

    def try_load_nodes(self, nodes: Union[PipelineNode, List[PipelineNode]], fold_id: Optional[int] = None):
        """
        :param nodes: nodes which fitted state should be loaded from cache
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        try:
            nodes_lst = ensure_wrapped_in_sequence(nodes)
            structural_ids = [_get_structural_id(node, fold_id) for node in nodes_lst]
            cached_ops = self._db.get_operations(structural_ids)
            for idx, cached_op in enumerate(cached_ops):
                if cached_op is not None:
                    nodes_lst[idx].fitted_operation = cached_op
                else:
                    nodes_lst[idx].fitted_operation = None
        except Exception as ex:
            self.log.warning(f'Cache can not be loaded: {ex}. Continue.', exc=ex, raise_if_test=True)

    def try_load_into_pipeline(self, pipeline: 'Pipeline', fold_id: Optional[int] = None):
        """
        :param pipeline: pipeline for loading into from cache
        :param fold_id: optional part of cache item UID (number of the CV fold)
        """
        self.try_load_nodes(pipeline.nodes, fold_id)


def _get_structural_id(node: PipelineNode, fold_id: Optional[int] = None) -> str:
    """
    Gets unique id from node.

    :param node: node to get uid from
    :param fold_id: fold number to fit data
    :return structural_id: unique node identificator
    """
    structural_id = node.descriptive_id
    structural_id += f'_{fold_id}' if fold_id is not None else ''
    return structural_id
