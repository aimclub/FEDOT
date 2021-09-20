import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np

from fedot.core.log import default_log
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.validation import validate
from fedot.core.utils import fedot_project_root
from fedot.remote.infrastructure.models_controller.computations import Client


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


@dataclass
class RemoteEvalParams:
    mode: str = 'remote',
    dataset_name: str = 'cholesterol',
    task_type: str = 'Task(TaskTypesEnum.regression)',
    train_data_idx: List = []
    is_multi_modal: bool = True
    var_names: List = []
    max_parallel: int = 7
    access_params: dict = {}


@singleton
class ComputationalSetup:
    remote_eval_params = {
        'mode': 'local',
        'dataset_name': '',  # name of the dataset for composer evaluation
        'task_type': '',  # name of the modelling task for dataset
        'train_data_idx': [],
        'max_parallel': 10
    }

    def __init__(self, remote_eval_params: RemoteEvalParams):
        """
        :param remote_eval_params: dictionary with the parameters of remote evaluation.
        """
        self._logger = default_log('RemoteFitterLog')
        self.remote_eval_params = remote_eval_params

    @property
    def is_remote(self):
        return ComputationalSetup.remote_eval_params['mode'] == 'remote'

    def fit(self, pipelines: List['Pipeline']) -> List['Pipeline']:
        remote_eval_params = ComputationalSetup.remote_eval_params

        dataset_name = remote_eval_params['dataset_name']
        task_type = remote_eval_params['task_type']
        data_idx = remote_eval_params['train_data_idx']

        var_names = remote_eval_params['var_names']
        is_multi_modal = remote_eval_params['is_multi_modal']

        if ('access_params' in remote_eval_params and
                remote_eval_params['access_params'] is not None):
            access_params = remote_eval_params['access_params']
        else:
            access_params = {
                'FEDOT_LOGIN': os.environ['FEDOT_LOGIN'],
                'FEDOT_PASSWORD': os.environ['FEDOT_PASSWORD'],
                'AUTH_SERVER': os.environ['AUTH_SERVER'],
                'CONTR_SERVER': os.environ['CONTR_SERVER'],
                'PROJECT_ID': os.environ['PROJECT_ID'],
                'DATA_ID': os.environ['DATA_ID']
            }
        client = _prepare_client(access_params)
        pipelines_parts, data_id = _prepare_computation_vars(pipelines, access_params)

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

                config = _get_config(pipeline_json, data_id, dataset_name,
                                     task_type, data_idx, var_names, is_multi_modal)

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
                        path=os.path.join(fedot_project_root(), 'remote_fit_results'),
                        unpack=True
                    )

                    try:
                        results_path_out = os.path.join(fedot_project_root(),
                                                        'remote_fit_results',
                                                        f'execution-{pipeline.execution_id}',
                                                        'out')
                        results_folder = os.listdir(results_path_out)[0]
                        pipeline.load(os.path.join(results_path_out, results_folder, 'fitted_pipeline.json'))
                    except Exception as ex:
                        self._logger.warn(f'{p_id}, {ex}')
            final_pipelines.extend(pipelines_part)

            self._logger.info(f'REMOTE EXECUTION TIME {end - start}')

        return final_pipelines


def _prepare_computation_vars(pipelines, access_params):
    num_parts = np.floor(len(pipelines) / ComputationalSetup.remote_eval_params['max_parallel'])
    num_parts = max(num_parts, 1)
    pipelines_parts = [x.tolist() for x in np.array_split(pipelines, num_parts)]
    data_id = int(access_params['DATA_ID'])
    return pipelines_parts, data_id


def _prepare_client(access_params):
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


def _get_config(pipeline_json, data_id, dataset_name, task_type, dataset_idx,
                var_names, is_multi_modal):
    return f"""[DEFAULT]
        pipeline_description = {pipeline_json}
        train_data = input_data_dir/data/{data_id}/{dataset_name}.csv
        task = {task_type}
        output_path = output_data_dir/fitted_pipeline
        train_data_idx = {[str(ind) for ind in dataset_idx]}
        var_names = {[str(name) for name in var_names]}
        is_multi_modal = {is_multi_modal}
        [OPTIONAL]
        """.encode('utf-8')
