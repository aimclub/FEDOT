import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.validation import validate
from fedot.utilities.pattern_wrappers import singleton
from fedot.remote.infrastructure.clients.client import Client


def init_data_for_remote_execution(train_data: InputData):
    setup = RemoteEvaluator()
    if setup.remote_task_params is not None:
        setup.remote_task_params.train_data_idx = train_data.idx


@dataclass
class RemoteTaskParams:
    """ Class with parameters of remote evaluation
    :param mode: evaluation mode - 'remote' or 'local'
    :param dataset_name: name of remote dataset used for fitting
    :param task_type: string representation of Task class for FEDOT
    :param train_data_idx: indices to subset dataset for fitting
    :param is_multi_modal: is train data multi-modal?
    :param var_names: variable names for fitting?
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


@singleton
class RemoteEvaluator:
    def __init__(self):
        """
        Class for the batch evaluation of pipelines using remote client
        """
        self._logger = default_log('RemoteFitterLog')
        self.remote_task_params = None
        self.client = None

    def init(self, client: Client = None, remote_task_params: Optional[RemoteTaskParams] = None):
        """
        :param client: client class for connection to external computational server.
        :param remote_task_params: dictionary with the parameters of remote evaluation.

        """
        self.remote_task_params = remote_task_params
        self.client = client

    @property
    def use_remote(self):
        return self.remote_task_params is not None and self.remote_task_params.mode == 'remote'

    def compute_pipelines(self, pipelines: List['Pipeline']) -> List['Pipeline']:
        params = self.remote_task_params

        client = self.client
        pipelines_parts = _prepare_batches(pipelines, params)
        final_pipelines = []

        # start of the remote execution for each pipeline
        for pipelines_part in pipelines_parts:
            for pipeline in pipelines_part:
                try:
                    validate(pipeline)
                except ValueError:
                    pipeline.execution_id = None
                    continue

                pipeline_json, _ = pipeline.save()
                pipeline_json = pipeline_json.replace('\n', '')

                config = _get_config(pipeline_json, params, self.client.exec_params, self.client.connect_params)

                task_id = client.create_task(config=config)

                pipeline.execution_id = task_id

            # waiting for readiness of all pipelines
            ex_time = client.wait_until_ready()

            # download of remote execution result for each pipeline
            for p_id, pipeline in enumerate(pipelines_part):
                if pipeline.execution_id:
                    try:
                        pipelines_part[p_id] = client.download_result(
                            execution_id=pipeline.execution_id
                        )
                    except Exception as ex:
                        self._logger.warn(f'{p_id}, {ex}')
            final_pipelines.extend(pipelines_part)

            self._logger.info(f'REMOTE EXECUTION TIME {ex_time}')

        return final_pipelines


def _prepare_batches(pipelines, params):
    num_parts = np.floor(len(pipelines) / params.max_parallel)
    num_parts = max(num_parts, 1)
    pipelines_parts = [x.tolist() for x in np.array_split(pipelines, num_parts)]
    return pipelines_parts


def _get_config(pipeline_json, params: RemoteTaskParams, client_params: dict, conn_params: dict):
    var_names = [str(name) for name in params.var_names] \
        if params.var_names is not None else []
    train_data_idx = [str(idx) for idx in params.train_data_idx] \
        if params.train_data_idx is not None else []

    data_name = params.dataset_name
    if conn_params is not None and len(conn_params) > 0:
        train_data = f"{client_params['container_input_path']}/data/{conn_params['DATA_ID']}/{data_name}.csv"
    else:
        train_data = f"{client_params['container_input_path']}/{data_name}.csv"
    return f"""[DEFAULT]
        pipeline_template = {pipeline_json}
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
