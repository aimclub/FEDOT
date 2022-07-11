from typing import TYPE_CHECKING, List, Optional, Union

from fedot.core.caching.base_cache import BaseCache
from fedot.core.caching.pipelines_cache_db import OperationsCacheDB
from fedot.core.pipelines.node import Node
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


class OperationsCache(BaseCache):
    """
    Stores/loads nodes `fitted_operation` field to increase performance of calculations.

    :param db_path: optional str determining a file name for caching pipelines
    """

    def __init__(self, db_path: Optional[str] = None):
        super().__init__(OperationsCacheDB(db_path))

    def save_nodes(self, nodes: Union[Node, List[Node]], fold_id: Optional[int] = None):
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
            self.log.warning(f'Nodes can not be saved: {ex}. Continue')

    def save_pipeline(self, pipeline: 'Pipeline', fold_id: Optional[int] = None):
        """
        :param pipeline: pipeline to be cached
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        self.save_nodes(pipeline.nodes, fold_id)

    def try_load_nodes(self, nodes: Union[Node, List[Node]], fold_id: Optional[int] = None) -> bool:
        """
        :param nodes: nodes which fitted state should be loaded from cache
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)

        :return cache_was_used: bool indicating if at least one item was loaded
        """
        cache_was_used = False
        try:
            nodes_lst = ensure_wrapped_in_sequence(nodes)
            structural_ids = [_get_structural_id(node, fold_id) for node in nodes_lst]
            cached_ops = self._db.get_operations(structural_ids)
            for idx, cached_op in enumerate(cached_ops):
                if cached_op is not None:
                    nodes_lst[idx].fitted_operation = cached_op
                    cache_was_used = True
                else:
                    nodes_lst[idx].fitted_operation = None
        except Exception as ex:
            self.log.warning(f'Cache can not be loaded: {ex}. Continue.')
        finally:
            return cache_was_used

    def try_load_into_pipeline(self, pipeline: 'Pipeline', fold_id: Optional[int] = None) -> bool:
        """
        :param pipeline: pipeline for loading into from cache
        :param fold_id: optional part of cache item UID
                            (number of the CV fold)

        :return: bool indicating if at least one item was loaded
        """
        return self.try_load_nodes(pipeline.nodes, fold_id)


def _get_structural_id(node: Node, fold_id: Optional[int] = None) -> str:
    """
    Gets unique id from node.

    :param node: node to get uid from
    :param fold_id: fold number to fit data
    :return structural_id: unique node identificator
    """
    structural_id = node.descriptive_id
    structural_id += f'_{fold_id}' if fold_id is not None else ''
    return structural_id
