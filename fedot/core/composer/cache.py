import glob
import os
import shelve
import uuid
from collections import namedtuple

from fedot.core.log import default_log
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

    def save_node(self, node, partial_id=''):
        """
        :param node: node for caching
        :param partial_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        if node.fitted_operation is not None:
            _save_cache_for_node(self.db_path, f'{node.descriptive_id}_{partial_id}',
                                 CachedState(node.fitted_operation))

    def save_pipeline(self, pipeline, partial_id=''):
        """
        :param pipeline: pipeline for caching
        :param partial_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        try:
            for node in pipeline.nodes:
                _save_cache_for_node(self.db_path, f'{node.descriptive_id}_{partial_id}',
                                     CachedState(node.fitted_operation))
        except Exception as ex:
            self.log.info(f'Cache can not be saved: {ex}. Continue.')

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


def _save_cache_for_node(db_path: str, structural_id: str,
                         cache_from_node: CachedState):
    if cache_from_node.operation is not None:
        # if node successfully fitted
        with shelve.open(db_path) as cache:
            cache[structural_id] = cache_from_node


def _load_cache_for_node(db_path: str, structural_id: str):
    with shelve.open(db_path) as cache:
        cached_state = cache.get(structural_id, None)

    return cached_state


def clear_folder(folder_path: str):
    """ Delete files from chosen folder """
    temp_files = glob.glob(folder_path)
    for file in temp_files:
        os.remove(file)
