import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Any, TypeVar, Callable, Hashable

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.core.utilities.serializable import Serializable
from fedot.core.optimisers.gp_comp.evaluation import DelegateEvaluator
from fedot.remote.infrastructure.clients.client import Client
from fedot.utilities.pattern_wrappers import singleton


def init_data_for_remote_execution(train_data: InputData):
    setup = RemoteEvaluator()
    if setup.remote_task_params is not None:
        setup.remote_task_params.train_data_idx = train_data.idx


@dataclass
class RemoteTaskParams:
    """Class with parameters of remote evaluation.

    :param mode: evaluation mode - 'remote' or 'local'
    :param dataset_name: name of remote dataset used for fitting
    :param task_type: string representation of Task class for FEDOT
    :param train_data_idx: indices to subset dataset for fitting
    :param is_multi_modal: is train data multi-modal?
    :param var_names: variable names for features
    :param target: variable name for target
    :param max_parallel maximal number of parallel remote task
    """
    mode: str = 'local'
    dataset_name: Optional[str] = None
    task_type: Optional[str] = None
    train_data_idx: Optional[List] = None
    is_multi_modal: bool = False
    var_names: Optional[List] = None
    target: Optional[str] = None
    max_parallel: int = 7


G = TypeVar('G', bound=Serializable)


@singleton
class RemoteEvaluator(DelegateEvaluator):
    def __init__(self):
        """
        Class for the batch evaluation of pipelines using remote client
        """
        self._logger = default_log(prefix='RemoteFitterLog')
        self.remote_task_params = None
        self.client = None
        self.config_for_dump = _get_config

    def init(self, client: Client = None,
             remote_task_params: Optional[RemoteTaskParams] = None,
             get_config: Optional[Callable] = None):
        """
        :param client: client class for connection to external computational server.
        :param remote_task_params: dictionary with the parameters of remote evaluation.
        :param get_config: optional function that constructs config for remote client.

        """
        self.remote_task_params = remote_task_params
        self.client = client
        self.config_for_dump = get_config or _get_config

    @property
    def is_enabled(self):
        return self.remote_task_params is not None and self.remote_task_params.mode == 'remote'

    def compute_graphs(self, graphs: Sequence[G]) -> Sequence[G]:
        params = self.remote_task_params

        client = self.client
        execution_ids = {}
        graph_batches = _prepare_batches(graphs, params.max_parallel)
        final_graphs = []

        # start of the remote execution for each pipeline
        for graphs_batch in graph_batches:
            for graph in graphs_batch:
                task_id = self._create_graph_task(graph)
                execution_ids[id(graph)] = task_id

            # waiting for readiness of all pipelines
            ex_time = client.wait_until_ready()

            # download of remote execution result for each pipeline
            for p_id, graph in enumerate(graphs_batch):
                task_id = execution_ids.get(id(graph), None)
                if task_id:
                    try:
                        graphs_batch[p_id] = client.download_result(task_id)
                    except Exception as ex:
                        self._logger.warning(f'{p_id}, {ex}')
            final_graphs.extend(graphs_batch)

            self._logger.info(f'REMOTE EXECUTION TIME {ex_time}')

        return final_graphs

    def _create_graph_task(self, graph: G) -> Optional[Hashable]:
        """Serializes task and creates a graph task for remote client.

        :return: task id
        """

        graph_json, _ = graph.save()
        graph_json = graph_json.replace('\n', '')

        config = self.config_for_dump(graph_json, self.remote_task_params,
                                      self.client.exec_params, self.client.connect_params)

        task_id = self.client.create_task(config=config)
        return task_id


def _prepare_batches(graphs: Sequence[Any], max_parallel: int):
    num_parts = np.floor(len(graphs) / max_parallel)
    num_parts = max(num_parts, 1)
    pipelines_parts = [x.tolist() for x in np.array_split(graphs, num_parts)]
    return pipelines_parts


def _get_config(graph_json: dict, params: RemoteTaskParams, client_params: dict, conn_params: dict):
    var_names = list(map(str, params.var_names)) \
        if params.var_names is not None else []
    train_data_idx = list(map(str, params.train_data_idx)) \
        if params.train_data_idx is not None else []

    data_name = params.dataset_name
    if conn_params is not None and conn_params:
        train_data = f"{client_params['container_input_path']}/data/{conn_params['DATA_ID']}/{data_name}.csv"
    else:
        train_data = f"{client_params['container_input_path']}/{data_name}.csv"
    return f"""[DEFAULT]
        pipeline_template = {graph_json}
        train_data = {train_data}
        task = {params.task_type}
        output_path = {client_params['container_output_path']}
        train_data_idx = {train_data_idx}
        var_names = {var_names}
        is_multi_modal = {params.is_multi_modal}
        target = {params.target}
        [OPTIONAL]
        """.encode('utf-8')


def _init_from_env():
    return {'FEDOT_LOGIN': os.environ['FEDOT_LOGIN'],
            'FEDOT_PASSWORD': os.environ['FEDOT_PASSWORD'],
            'AUTH_SERVER': os.environ['AUTH_SERVER'],
            'CONTR_SERVER': os.environ['CONTR_SERVER'],
            'PROJECT_ID': os.environ['PROJECT_ID'],
            'DATA_ID': os.environ['DATA_ID']}
