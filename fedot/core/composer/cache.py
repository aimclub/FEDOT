import glob
import os
import shelve
import uuid

from collections import namedtuple
from typing import List, Union

from fedot.core.log import default_log
from fedot.core.pipelines.node import Node
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

    def save_nodes(self, nodes: Union[Node, List[Node]], partial_id=''):
        """
        :param nodes: node/nodes for caching
        :param partial_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        try:
            with shelve.open(self.db_path) as cache:
                if not isinstance(nodes, list):
                    nodes = [nodes]
                for node in nodes:
                    _save_cache_for_node(cache, node, partial_id)
        except Exception as ex:
            self.log.info(f'Nodes can not be saved: {ex}. Continue')

    def save_pipeline(self, pipeline: Pipeline, partial_id=''):
        """
        :param pipeline: pipeline for caching
        :param partial_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        self.save_nodes(pipeline.nodes, partial_id)

    def clear(self, tmp_only=False):
        if not tmp_only:
            for ext in ['bak', 'dir', 'dat']:
                if os.path.exists(f'{self.db_path}.{ext}'):
                    os.remove(f'{self.db_path}.{ext}')
        folder_path = f'{str(default_fedot_data_dir())}/tmp_*'
        clear_folder(folder_path)

    def get(self, node, partial_id=''):
        """
        :param node: node which fitted state should be loaded from cache
        :param partial_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        found_operation = _load_cache_for_node(self.db_path,
                                               f'{node.descriptive_id}_{partial_id}')
        return found_operation


def _save_cache_for_node(cache_shelf: shelve.Shelf, node: Node, partial_id=''):
    cached_state = CachedState(node.fitted_operation)
    if cached_state.operation is not None:
        structural_id = f'{node.descriptive_id}_{partial_id}'
        cache_shelf[structural_id] = cached_state


def _load_cache_for_node(db_path: str, structural_id: str):
    with shelve.open(db_path) as cache:
        cached_state = cache.get(structural_id, None)

    return cached_state


def clear_folder(folder_path: str):
    """ Delete files from chosen folder """
    temp_files = glob.glob(folder_path)
    for file in temp_files:
        os.remove(file)
