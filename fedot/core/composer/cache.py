import shelve
import shutil
import uuid
from collections import namedtuple
from multiprocessing import RLock, Manager
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union, Type

from fedot.core.log import Log, SingletonMeta, default_log
from fedot.core.pipelines.node import Node
from fedot.core.utilities.data_structures import ensure_list

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline

from fedot.core.utils import default_fedot_data_dir

CachedState = namedtuple('CachedState', 'operation')


class OperationsCache(metaclass=SingletonMeta):
    _rlock = RLock()

    def __init__(self, log: Optional[Log] = None, db_path: Optional[str] = None, clear_exiting=True):
        self.log = log or default_log(__name__)
        self.db_path = db_path or Path(str(default_fedot_data_dir()), f'tmp_{str(uuid.uuid4())}').as_posix()

        self._utility = Manager().dict(
            dict.fromkeys(['pipelines_loaded', 'nodes_loaded', 'pipelines_passed', 'nodes_passed'], 0)
        )

        if clear_exiting:
            self.clear()

    def reset(self):
        for k in self._utility:
            self._utility[k] = 0
        self.clear()

    @property
    def effectiveness(self):
        pipelines_passed = self._utility['pipelines_passed']
        nodes_passed = self._utility['nodes_passed']

        return {
            'pipelines': round(self._utility['pipelines_loaded'] / pipelines_passed, 3) if pipelines_passed else 0.,
            'nodes': round(self._utility['nodes_loaded'] / nodes_passed, 3) if nodes_passed else 0.
        }

    def save_nodes(self, nodes: Union[Node, List[Node]], fold_id: Optional[int] = None):
        """
        :param nodes: node/nodes for caching
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        with OperationsCache._rlock:
            try:
                with shelve.open(self.db_path) as cache:
                    for node in ensure_list(nodes):
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

    def try_load_nodes(self, nodes: Union[Node, List[Node]], fold_id: Optional[int] = None) -> bool:
        """
        :param nodes: nodes which fitted state should be loaded from cache
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        with OperationsCache._rlock:
            cache_was_used = False
            try:
                with shelve.open(self.db_path) as cache:
                    for node in ensure_list(nodes):
                        cached_state = _load_cache_for_node(cache, node, fold_id)
                        if cached_state is not None:
                            node.fitted_operation = cached_state.operation
                            cache_was_used = True
                            self._utility['nodes_loaded'] += 1
                        else:
                            node.fitted_operation = None
                        self._utility['nodes_passed'] += 1
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
        with OperationsCache._rlock:
            loaded_before = self._utility['nodes_loaded']
            did_load_any = self.try_load_nodes(pipeline.nodes, fold_id)
            loaded_after = self._utility['nodes_loaded']

            if loaded_after - loaded_before == len(pipeline.nodes):
                self._utility['pipelines_loaded'] += 1
            self._utility['pipelines_passed'] += 1

            return did_load_any

    def clear(self, tmp_only=False):
        if not tmp_only:
            for ext in ['bak', 'dir', 'dat']:
                file = Path(f'{self.db_path}.{ext}')
                if file.exists():
                    file.unlink()
        _clear_from_temporaries(default_fedot_data_dir())


def _get_structural_id(node: Node, fold_id: Optional[int] = None) -> str:
    structural_id = node.descriptive_id
    structural_id += f'_{fold_id}' if fold_id is not None else ''
    return structural_id


def _save_cache_for_node(cache_shelf: shelve.Shelf, node: Node, fold_id: Optional[int] = None):
    if node.fitted_operation is not None:
        cached_state = CachedState(node.fitted_operation)
        structural_id = _get_structural_id(node, fold_id)
        cache_shelf[structural_id] = cached_state


def _load_cache_for_node(cache_shelf: shelve.Shelf,
                         node: Node, fold_id: Optional[int] = None) -> Optional[Type[CachedState]]:
    structural_id = _get_structural_id(node, fold_id)
    cached_state = cache_shelf.get(structural_id, None)

    return cached_state


def _clear_from_temporaries(folder_path: str):
    """ Deletes temporary files from chosen folder """
    for file in Path(folder_path).glob('tmp_*'):
        if file.is_dir():
            shutil.rmtree(file)
        else:
            file.unlink()
