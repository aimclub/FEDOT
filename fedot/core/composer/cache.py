import os
import shelve
import uuid
from collections import namedtuple

from fedot.core.utils import default_fedot_data_dir

CachedState = namedtuple('CachedState', 'operation')


class OperationsCache:
    def __init__(self, db_path=None, clear_exiting=True):
        if not db_path:
            self.db_path = f'{str(default_fedot_data_dir())}/{str(uuid.uuid4())}'
        else:
            self.db_path = db_path

        if clear_exiting:
            self.clear()

    def save_node(self, node):
        if node.fitted_operation is not None:
            _save_cache_for_node(self.db_path, node.descriptive_id,
                                 CachedState(node.fitted_operation))

    def save_chain(self, chain):
        for node in chain.nodes:
            _save_cache_for_node(self.db_path, node.descriptive_id,
                                 CachedState(node.fitted_operation))

    def clear(self):
        for ext in ['bak', 'dir', 'dat']:
            if os.path.exists(f'{self.db_path}.{ext}'):
                os.remove(f'{self.db_path}.{ext}')

    def get(self, node):
        found_operation = _load_cache_for_node(self.db_path, node.descriptive_id)
        # TODO: Add node and node from cache "fitted on data" field comparison
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
