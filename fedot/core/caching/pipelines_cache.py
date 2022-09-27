from typing import List, Optional, TYPE_CHECKING, Union

from fedot.core.caching.base_cache import BaseCache
from fedot.core.caching.pipelines_cache_db import OperationsCacheDB
from fedot.core.pipelines.node import Node
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence
from fedot.utilities.debug import is_test_session

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


class OperationsCache(BaseCache):
    """
    Stores/loads nodes `fitted_operation` field to increase performance of calculations.

    :param cache_folder: path to the place where cache files should be stored.
    """

    def __init__(self, cache_folder: Optional[str] = None):
        super().__init__(OperationsCacheDB(cache_folder))

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
            if is_test_session():
                raise ex

    def save_pipeline(self, pipeline: 'Pipeline', fold_id: Optional[int] = None):
        """
        :param pipeline: pipeline to be cached
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        self.save_nodes(pipeline.nodes, fold_id)

    def try_load_nodes(self, nodes: Union[Node, List[Node]], fold_id: Optional[int] = None):
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
            self.log.warning(f'Cache can not be loaded: {ex}. Continue.')
            if is_test_session():
                raise ex

    def try_load_into_pipeline(self, pipeline: 'Pipeline', fold_id: Optional[int] = None):
        """
        :param pipeline: pipeline for loading into from cache
        :param fold_id: optional part of cache item UID (number of the CV fold)
        """
        self.try_load_nodes(pipeline.nodes, fold_id)


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
