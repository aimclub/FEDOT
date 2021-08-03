import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.validation import validate
from fedot.core.utils import default_fedot_data_dir
from fedot.remote.infrastructure.models_controller.computations import Client


def init_data_for_remote_execution(train_data: InputData):
    setup = ComputationalSetup()
    if setup.remote_eval_params is not None:
        setup.remote_eval_params.train_data_idx = train_data.idx


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


@dataclass
class RemoteEvalParams:
    """ Class with parameters of remote evaluation
    :param mode: evaluation mode - 'remote' or 'local'
    :param dataset_name: name of remote dataset used for fitting
    :param task_type: string representation of Task class for FEDOT
    :param train_data_idx: indices to subset dataset for fitting
    :param is_multi_modal: is train data multi-modal?
    :param var_names: variable names for fitting?
    :param max_parallel maximal number of parallel remote task
    :param access_params optional set of parameters for remote server connection.
        If None, the environmental variables are used.
    """
    mode: str
    dataset_name: str
    task_type: str
    train_data_idx: Optional[List] = None
    is_multi_modal: bool = False
    var_names: Optional[List] = None
    max_parallel: int = 7
    access_params: Optional[dict] = None


@singleton
class ComputationalSetup:
    def __init__(self, remote_eval_params: Optional[RemoteEvalParams] = None):
        """
        :param remote_eval_params: dictionary with the parameters of remote evaluation.
        """
        self._logger = default_log('RemoteFitterLog')
        self.remote_eval_params = remote_eval_params

    @property
    def is_remote(self):
        return self.remote_eval_params is not None and self.remote_eval_params['mode'] == 'remote'

    def fit(self, pipelines: List['Pipeline']) -> List['Pipeline']:
        params = self.remote_eval_params

        if params.access_params is not None:
            params.access_params = params.access_params
        else:
            params.access_params = _init_from_env()
        client = _prepare_client(params)
        pipelines_parts, data_id = _prepare_computation_vars(pipelines, params)

        final_pipelines = []
        for pipelines_part in pipelines_parts:
            for pipeline in pipelines_part:
                try:
                    validate(pipeline)
                except ValueError:
                    pipeline.execution_id = None
                    continue

                pipeline_json, _ = pipeline.save()
                pipeline_json = pipeline_json.replace('\n', '')

                config = _get_config(pipeline_json, data_id, params)

                created_ex = client.create_execution(
                    container_input_path="/home/FEDOT/input_data_dir",
                    container_output_path="/home/FEDOT/output_data_dir",
                    container_config_path="/home/FEDOT/.config",
                    container_image="fedot:dm-6",
                    timeout=360,
                    config=config
                )

                pipeline.execution_id = created_ex['id']

            statuses = ['']
            all_executions = client.get_executions()
            self._logger.info(all_executions)
            start = datetime.now()
            while any(s not in ['Succeeded', 'Failed', 'Timeout', 'Interrupted'] for s in statuses):
                executions = client.get_executions()
                statuses = [execution['status'] for execution in executions]
                self._logger.info([f"{execution['id']}={execution['status']};" for execution in executions])
                time.sleep(5)

            end = datetime.now()

            for p_id, pipeline in enumerate(pipelines_part):
                if pipeline.execution_id:
                    client.download_result(
                        execution_id=pipeline.execution_id,
                        path=os.path.join(default_fedot_data_dir(), 'remote_fit_results'),
                        unpack=True
                    )

                    try:
                        results_path_out = os.path.join(default_fedot_data_dir(),
                                                        'remote_fit_results',
                                                        f'execution-{pipeline.execution_id}',
                                                        'out')
                        results_folder = os.listdir(results_path_out)[0]
                        pipeline.load(os.path.join(results_path_out, results_folder, 'fitted_pipeline.json'))

                        for root, dirs, files in os.walk(results_path_out):
                            for file in files:
                                os.remove(os.path.join(root, file))
                    except Exception as ex:
                        self._logger.warn(f'{p_id}, {ex}')
            final_pipelines.extend(pipelines_part)

            self._logger.info(f'REMOTE EXECUTION TIME {end - start}')

        return final_pipelines


def _prepare_computation_vars(pipelines, params):
    access_params = params.access_params
    num_parts = np.floor(len(pipelines) / params.max_parallel)
    num_parts = max(num_parts, 1)
    pipelines_parts = [x.tolist() for x in np.array_split(pipelines, num_parts)]
    data_id = int(access_params['DATA_ID'])
    return pipelines_parts, data_id


def _prepare_client(params):
    access_params = params.access_params
    pid = int(access_params['PROJECT_ID'])

    client = Client(
        authorization_server=access_params['AUTH_SERVER'],
        controller_server=access_params['CONTR_SERVER']
    )

    client.login(login=access_params['FEDOT_LOGIN'],
                 password=access_params['FEDOT_PASSWORD'])

    group = client.create_execution_group(project_id=pid)
    client.set_group_token(project_id=pid, group_id=group['id'])
    return client


def _get_config(pipeline_json, data_id, params: RemoteEvalParams):
    var_names = [str(name) for name in params.var_names] \
        if params.var_names is not None else []
    train_data_idx = [str(idx) for idx in params.train_data_idx] \
        if params.train_data_idx is not None else []

    return f"""[DEFAULT]
        pipeline_description = {pipeline_json}
        train_data = input_data_dir/data/{data_id}/{params.dataset_name}.csv
        task = {params.task_type}
        output_path = output_data_dir/fitted_pipeline
        train_data_idx = {train_data_idx}
        var_names = {var_names}
        is_multi_modal = {params.is_multi_modal}
        [OPTIONAL]
        """.encode('utf-8')


def _init_from_env():
    return {'FEDOT_LOGIN': os.environ['FEDOT_LOGIN'],
            'FEDOT_PASSWORD': os.environ['FEDOT_PASSWORD'],
            'AUTH_SERVER': os.environ['AUTH_SERVER'],
            'CONTR_SERVER': os.environ['CONTR_SERVER'],
            'PROJECT_ID': os.environ['PROJECT_ID'],
            'DATA_ID': os.environ['DATA_ID']}
