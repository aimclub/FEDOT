import pickle
import re
import sqlite3
import string
import uuid

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, TypeVar, Union

from fedot.core.log import Log, SingletonMeta, default_log
from fedot.core.operations.operation import Operation
from fedot.core.pipelines.node import Node
from fedot.core.utilities.data_structures import ensure_list
from fedot.core.utils import default_fedot_data_dir

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline

IOperation = TypeVar('IOperation', bound=Operation)


@dataclass
class CachedState:
    operation: IOperation


class OperationsCache(metaclass=SingletonMeta):
    """
    Stores/loads nodes `fitted_operation` field to increase performance of calculations.

    :param log: optional Log object to record messages
    :param db_path: optional str determining a file name for caching pipelines
    """

    def __init__(self, log: Optional[Log] = None, db_path: Optional[str] = None):
        self.log = log or default_log(__name__)
        self.db_path = db_path or Path(str(default_fedot_data_dir()), f'tmp_{str(uuid.uuid4())}').as_posix()

        self._del_temps()

        self._effectiveness_keys = ['pipelines_hit', 'nodes_hit', 'pipelines_total', 'nodes_total']
        self._eff_table = 'effectiveness'
        self._op_table = 'operations'
        self._init_db()

    @property
    def effectiveness_ratio(self):
        """
        Returns percent of how many pipelines/nodes were loaded instead of computing
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            #  Result order corresponds to the order in self._effectiveness_keys
            pipelines_hit, nodes_hit, pipelines_total, nodes_total = self._get_eff(conn)

            return {
                'pipelines': round(pipelines_hit / pipelines_total, 3) if pipelines_total else 0.,
                'nodes': round(nodes_hit / nodes_total, 3) if nodes_total else 0.
            }

    def reset(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            self._reset_eff(conn)
            self._reset_ops(conn)

    def save_nodes(self, nodes: Union[Node, List[Node]], fold_id: Optional[int] = None):
        """
        :param nodes: node/nodes for caching
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                for node in ensure_list(nodes):
                    self._add_op(conn, node, fold_id)
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
        cache_was_used = False
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                for node in ensure_list(nodes):
                    cached_state = self._get_op(conn, node, fold_id)
                    if cached_state is not None:
                        node.fitted_operation = cached_state.operation
                        cache_was_used = True
                        self._inc_eff(conn, 'nodes_hit')
                    else:
                        node.fitted_operation = None
                    self._inc_eff(conn, 'nodes_total')
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
        with closing(sqlite3.connect(self.db_path)) as conn:
            _, hits_before, _, _ = self._get_eff(conn)
        did_load_any = self.try_load_nodes(pipeline.nodes, fold_id)
        with closing(sqlite3.connect(self.db_path)) as conn:
            _, hits_after, _, _ = self._get_eff(conn)

            if hits_after - hits_before >= len(pipeline.nodes):  # TODO: some kind of heuristic, not too accurate
                self._inc_eff(conn, 'pipelines_hit')
            self._inc_eff(conn, 'pipelines_total')

            return did_load_any

    def _del_temps(self):
        db_path = Path(self.db_path)
        for file in db_path.parent.glob('tmp_*'):
            file.unlink()

    def _init_db(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                eff_type = ' INTEGER DEFAULT 0'
                fields = f'{eff_type},'.join(self._effectiveness_keys) + eff_type
                cur.execute((
                    f'CREATE TABLE IF NOT EXISTS {self._eff_table} ('
                        'id INTEGER PRIMARY KEY CHECK (id = 1),'
                        f'{fields}'
                    ');'
                ))
                cur.execute(f'INSERT INTO {self._eff_table} DEFAULT VALUES;')
            with conn:
                cur = conn.cursor()
                cur.execute((
                    f'CREATE TABLE IF NOT EXISTS {self._op_table} ('
                        'id TEXT PRIMARY KEY,'
                        'operation BLOB'
                    ');'
                ))

    def _get_eff(self, conn: sqlite3.Connection) -> Tuple[int, int, int, int]:
        with conn:
            cur = conn.cursor()
            cur.execute(f'SELECT {",".join(self._effectiveness_keys)} FROM {self._eff_table}')
            return cur.fetchone()

    def _inc_eff(self, conn: sqlite3.Connection, col: str):
        with conn:
            cur = conn.cursor()
            cur.execute(f'UPDATE {self._eff_table} SET {col} = {col} + 1')

    def _reset_eff(self, conn: sqlite3.Connection):
        with conn:
            cur = conn.cursor()
            cur.execute(f'DELETE FROM {self._eff_table};')
            cur.execute(f'INSERT INTO {self._eff_table} DEFAULT VALUES;')

    def _reset_ops(self, conn: sqlite3.Connection):
        with conn:
            cur = conn.cursor()
            cur.execute(f'DELETE FROM {self._op_table}')

    def _get_op(self, conn: sqlite3.Connection, node: Node, fold_id: Optional[int] = None) -> Optional[CachedState]:
        with conn:
            cur = conn.cursor()
            structural_id = _get_structural_id(node, fold_id)
            cur.execute(f'SELECT operation FROM {self._op_table} WHERE id = ?', [structural_id])
            retrieved = cur.fetchone()
            if retrieved is not None:
                retrieved = pickle.loads(retrieved[0])
            return retrieved

    def _add_op(self, conn: sqlite3.Connection, node: Node, fold_id: Optional[int] = None):
        if node.fitted_operation is not None:
            with conn:
                cur = conn.cursor()
                cached_state = CachedState(node.fitted_operation)
                structural_id = _get_structural_id(node, fold_id)
                pdata = pickle.dumps(cached_state, pickle.HIGHEST_PROTOCOL)
                cur.execute(f'INSERT OR IGNORE INTO {self._op_table} VALUES (?, ?)',
                            [structural_id, sqlite3.Binary(pdata)])

    def __len__(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(f'SELECT id FROM {self._op_table}')
                all_rows = cur.fetchall()
                return len(all_rows)


def _get_structural_id(node: Node, fold_id: Optional[int] = None) -> str:
    structural_id = re.sub(f'[{string.punctuation}]+', '', node.descriptive_id)
    structural_id += f'_{fold_id}' if fold_id is not None else ''
    return structural_id
