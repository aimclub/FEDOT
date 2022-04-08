import shelve
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, TypeVar, Union, Type

from fedot.core.log import Log, SingletonMeta, default_log
from fedot.core.operations.operation import Operation
from fedot.core.pipelines.node import Node
from fedot.core.utilities.data_structures import ensure_list

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline

from fedot.core.utils import default_fedot_data_dir
from contextlib import nullcontext, contextmanager
from multiprocessing.managers import SyncManager

IOperation = TypeVar('IOperation', bound=Operation)


@dataclass
class CachedState:
    operation: IOperation


class OperationsCache(metaclass=SingletonMeta):
    '''
    Stores/loades nodes `fitted_operation` field to increase performance of calculations.

    :param mp_manager: optional multiprocessing manager in case of main API `n_jobs` != 1,
        used to synchronize access to class variables
    :param log: optional Log object to record messages
    :param db_path: optional str determining a file name for caching pipelines
    :param clear_exiting: optional bool indicating if it is needed to clean up resources before class can be used
    '''

    def __init__(self, mp_manager: Optional[SyncManager] = None, log: Optional[Log] = None,
                 db_path: Optional[str] = None,
                 clear_exiting=True):
        effectiveness_keys = ['pipelines_hit', 'nodes_hit', 'pipelines_total', 'nodes_total']
        if mp_manager is None:
            self._rlock = nullcontext()
            self._effectiveness = dict.fromkeys(effectiveness_keys, 0)
        else:
            self._rlock = mp_manager.RLock()
            self._effectiveness = mp_manager.dict(dict.fromkeys(effectiveness_keys, 0))

        self.log = log or default_log(__name__)
        self.db_path = db_path or Path(str(default_fedot_data_dir()), f'tmp_{str(uuid.uuid4())}').as_posix()

        if clear_exiting:
            self.clear()

    def reset(self):
        with self._rlock:
            for k in self._effectiveness:
                self._effectiveness[k] = 0
            self.clear()

    @contextmanager
    def using_resources(self):
        self.clear()
        try:
            yield
        finally:
            self.clear()

    @property
    def effectiveness_ratio(self):
        pipelines_hit = self._effectiveness['pipelines_hit']
        pipelines_total = self._effectiveness['pipelines_total']
        nodes_hit = self._effectiveness['nodes_hit']
        nodes_total = self._effectiveness['nodes_total']

        return {
            'pipelines': round(pipelines_hit / pipelines_total, 3) if pipelines_total else 0.,
            'nodes': round(nodes_hit / nodes_total, 3) if nodes_total else 0.
        }

    def save_nodes(self, nodes: Union[Node, List[Node]], fold_id: Optional[int] = None):
        """
        :param nodes: node/nodes for caching
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        with self._rlock:
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
        with self._rlock:
            cache_was_used = False
            try:
                with shelve.open(self.db_path) as cache:
                    for node in ensure_list(nodes):
                        cached_state = _load_cache_for_node(cache, node, fold_id)
                        if cached_state is not None:
                            node.fitted_operation = cached_state.operation
                            cache_was_used = True
                            self._effectiveness['nodes_hit'] += 1
                        else:
                            node.fitted_operation = None
                        self._effectiveness['nodes_total'] += 1
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
        with self._rlock:
            hits_before = self._effectiveness['nodes_hit']
            did_load_any = self.try_load_nodes(pipeline.nodes, fold_id)
            hits_after = self._effectiveness['nodes_hit']

            if hits_after - hits_before == len(pipeline.nodes):
                self._effectiveness['pipelines_hit'] += 1
            self._effectiveness['pipelines_total'] += 1

            return did_load_any

    def clear(self, tmp_only=False):
        with self._rlock:
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
