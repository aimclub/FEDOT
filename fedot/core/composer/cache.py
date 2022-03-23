import glob
import os
import shelve
import uuid
from collections import namedtuple
from typing import TYPE_CHECKING, List, Optional, Union

from fedot.core.log import default_log
from fedot.core.pipelines.node import Node

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline

from fedot.core.utils import default_fedot_data_dir

CachedState = namedtuple('CachedState', 'operation')


class OperationsCache:
    def __init__(self, log=None, db_path=None, clear_exiting=True):
        self.log = default_log(__name__) if log is None else log

        if not db_path:
            self.db_path = f'{str(default_fedot_data_dir())}/tmp_{str(uuid.uuid4())}'
        else:
            self.db_path = db_path

        if clear_exiting:
            self.clear()

    def save_nodes(self, nodes: Union[Node, List[Node]], fold_id: Optional[int] = None):
        """
        :param nodes: node/nodes for caching
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        try:
            with shelve.open(self.db_path) as cache:
                if not isinstance(nodes, list):
                    nodes = [nodes]
                for node in nodes:
                    _save_cache_for_node(cache, node, fold_id)
        except Exception as ex:
            self.log.info(f'Nodes can not be saved: {ex}. Continue')

    def save_pipeline(self, pipeline: 'Pipeline', fold_id: Optional[int] = None):
        """
        :param pipeline: pipeline for caching
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        self.save_nodes(pipeline.nodes, fold_id)

    def clear(self, tmp_only=False):
        if not tmp_only:
            for ext in ['bak', 'dir', 'dat']:
                if os.path.exists(f'{self.db_path}.{ext}'):
                    os.remove(f'{self.db_path}.{ext}')
        folder_path = f'{str(default_fedot_data_dir())}/tmp_*'
        clear_folder(folder_path)

    def get(self, node, fold_id: Optional[int] = None):
        """
        :param node: node which fitted state should be loaded from cache
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        found_operation = _load_cache_for_node(self.db_path, node, fold_id)
        return found_operation


def _get_structural_id(node: Node, fold_id: Optional[int] = None):
    structural_id = node.descriptive_id
    structural_id += f'_{fold_id}' if fold_id is not None else ''
    return structural_id


def _save_cache_for_node(cache_shelf: shelve.Shelf, node: Node, fold_id: Optional[int] = None):
    cached_state = CachedState(node.fitted_operation)
    if cached_state.operation is not None:
        structural_id = _get_structural_id(node, fold_id)
        cache_shelf[structural_id] = cached_state


def _load_cache_for_node(db_path: str, node: Node, fold_id: Optional[int] = None):
    with shelve.open(db_path) as cache:
        structural_id = _get_structural_id(node, fold_id)
        cached_state = cache.get(structural_id, None)

    return cached_state


def clear_folder(folder_path: str):
    """ Delete files from chosen folder """
    temp_files = glob.glob(folder_path)
    for file in temp_files:
        os.remove(file)
