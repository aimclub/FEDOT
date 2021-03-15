import shelve
from collections import namedtuple

CachedState = namedtuple('CachedState', 'preprocessor model')


class ModelsCache:
    def __init__(self, db_path='cache_db'):
        self.db_path = db_path
        self.clear()

    def save_node(self, node):
        if node.fitted_model is not None:
            _save_cache_for_node(self.db_path, node.descriptive_id,
                                 CachedState(node.fitted_preprocessor,
                                             node.fitted_model))

    def save_chain(self, chain):
        for node in chain.nodes:
            _save_cache_for_node(self.db_path, node.descriptive_id,
                                 CachedState(node.fitted_preprocessor,
                                             node.fitted_model))

    def clear(self):
        # if os.path.exists(self.db_path):
        #    os.remove(self.db_path)
        pass

    def get(self, node):
        found_model = _load_cache_for_node(self.db_path, node.descriptive_id)
        return found_model


def _save_cache_for_node(db_path: str, structural_id: str, node):
    with shelve.open(db_path) as cache:
        cache[structural_id] = node


def _load_cache_for_node(db_path: str, structural_id: str):
    with shelve.open(db_path) as cache:
        cached_state = cache.get(structural_id, None)

    return cached_state
