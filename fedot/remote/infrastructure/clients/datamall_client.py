import json
import os
import shutil
import time
from datetime import datetime, timedelta
from typing import List, Optional

import requests

from fedot.core.pipelines.pipeline import Pipeline
from fedot.remote.infrastructure.clients.client import Client

USER_TOKEN_KEY = 'x-jwt-auth'
GROUP_TOKEN_KEY = 'x-jwt-models-execution'

DEFAULT_EXEC_PARAMS = {
    'container_input_path': "/home/FEDOT/input_data_dir",
    'container_output_path': "/home/FEDOT/output_data_dir",
    'container_config_path': "/home/FEDOT/.config",
    'container_image': "fedot:dm-9",
    'timeout': 360
}

# example of connection params for DataMall
DEFAULT_CONNECT_PARAMS = {
    'FEDOT_LOGIN': 'fedot',
    'FEDOT_PASSWORD': 'fedot-password',
    'AUTH_SERVER': 'http://10.32.0.51:30880/b',
    'CONTR_SERVER': 'http://10.32.0.51:30880/models-controller',
    'PROJECT_ID': '83',
    'DATA_ID': '60'
}


# TO BE MOVED TO PYPI AS EXTERNAL LIB

class DataMallClient(Client):
    def __init__(self, connect_params: dict, exec_params: dict, output_path: Optional[str] = None):
        authorization_server = connect_params['AUTH_SERVER']
        controller_server = connect_params['CONTR_SERVER']
        self.authorization_server = os.environ['AUTH_SERVER'] if authorization_server is None else authorization_server
        self.controller_server = os.environ['CONTR_SERVER'] if controller_server is None else controller_server
        self._user_token = None
        self.user = None
        self._current_project_id = None
        self.group_token = None

        self._login(login=connect_params['FEDOT_LOGIN'],
                    password=connect_params['FEDOT_PASSWORD'])

        pid = connect_params['PROJECT_ID']
        group = self._create_execution_group(project_id=pid)
        self._set_group_token(project_id=pid, group_id=group['id'])

        super().__init__(connect_params, exec_params, output_path)

    def create_task(self, config) -> str:
        data_id = self.connect_params['DATA_ID']
        created_ex = self._create_execution(f"{self.exec_params['container_input_path']}",
                                            self.exec_params['container_output_path'],
                                            self.exec_params['container_config_path'],
                                            self.exec_params['container_image'],
                                            self.exec_params['timeout'],
                                            config=config)
        return created_ex['id']

    def wait_until_ready(self) -> timedelta:
        statuses = ['']
        all_executions = self._get_executions()
        self._logger.info(all_executions)
        start = datetime.now()
        while any(s not in ['Succeeded', 'Failed', 'Timeout', 'Interrupted'] for s in statuses):
            executions = self._get_executions()
            statuses = [execution['status'] for execution in executions]
            self._logger.info([f"{execution['id']}={execution['status']};" for execution in executions])
            time.sleep(5)
        end = datetime.now()
        ex_time = end - start
        return ex_time

    def _login(self, login: str, password: str) -> None:
        response = requests.post(
            url=f'{self.authorization_server}/users/login',
            json={
                'login': login,
                'password': password
            }
        )

        if response.status_code != 200:
            raise ValueError(f'Unable to get user token. Reason: {response.text}')

        self._user_token = response.cookies['x-jwt-auth']
        self.user = json.loads(response.text)

    def _set_group_token(self, project_id: int, group_token: str = None, group_id: int = None) -> None:
        if group_token is not None:
            self.group_token = group_token
            self._current_project_id = project_id
            return

        if group_id is not None:
            group = self._get_execution_group(
                project_id=project_id,
                group_id=group_id
            )

            self.group_token = group['token']
            self._current_project_id = group['project_id']

            return

        raise ValueError(f'You have to specify project_id and token/group_id!')

    def _get_execution_groups(self, project_id: int) -> List[dict]:
        response = requests.get(
            url=f'{self.controller_server}/execution-groups/{project_id}',
            cookies={USER_TOKEN_KEY: self._user_token}
        )

        if response.status_code != 200:
            raise ValueError(f'Unable to get execution groups. Reason: {response.text}')

        return json.loads(response.text)

    def _get_execution_group(self, project_id: int, group_id: int) -> dict:
        response = requests.get(
            url=f'{self.controller_server}/execution-groups/{project_id}/{group_id}',
            cookies={USER_TOKEN_KEY: self._user_token}
        )

        if response.status_code != 200:
            raise ValueError(f'Unable to get execution group. Reason: {response.text}')

        return json.loads(response.text)

    def _create_execution_group(self, project_id: int) -> dict:
        response = requests.post(
            url=f'{self.controller_server}/execution-groups/{project_id}',
            cookies={USER_TOKEN_KEY: self._user_token}
        )

        if response.status_code != 200:
            raise ValueError(f'Unable to create execution group. Reason: {response.text}')

        return json.loads(response.text)

    def _get_executions(self):
        response = requests.get(
            url=f'{self.controller_server}/executions/{self._current_project_id}',
            cookies={USER_TOKEN_KEY: self._user_token},
            headers={GROUP_TOKEN_KEY: self.group_token}
        )

        if response.status_code != 200:
            raise ValueError(f'Unable to get executions. Reason: {response.text}')

        return json.loads(response.text)

    def _get_execution(self, execution_id: int):
        response = requests.get(
            url=f'{self.controller_server}/executions/{self._current_project_id}/{execution_id}',
            cookies={USER_TOKEN_KEY: self._user_token},
            headers={GROUP_TOKEN_KEY: self.group_token}
        )

        if response.status_code != 200:
            raise ValueError(f'Unable to get execution. Reason: {response.text}')

        return json.loads(response.text)

    def _create_execution(self, container_input_path: str,
                          container_output_path: str,
                          container_config_path: str,
                          container_image: str,
                          timeout: int,
                          config: bytes) -> dict:
        response = requests.post(
            url=f'{self.controller_server}/executions/{self._current_project_id}',
            cookies={USER_TOKEN_KEY: self._user_token},
            headers={GROUP_TOKEN_KEY: self.group_token},
            files={
                'input_path': (None, container_input_path),
                'output_path': (None, container_output_path),
                'config_path': (None, container_config_path),
                'image': (None, container_image),
                'timeout': (None, timeout),
                'config_file': ('config', config)
            }
        )

        if response.status_code != 200:
            raise ValueError(f'Unable to create execution. Reason: {response.text}')

        return json.loads(response.text)

    def _stop_execution(self, execution_id: int) -> None:
        response = requests.delete(
            url=f'{self.controller_server}/executions/{self._current_project_id}/{execution_id}',
            cookies={USER_TOKEN_KEY: self._user_token},
            headers={GROUP_TOKEN_KEY: self.group_token}
        )

        if response.status_code != 204:
            raise ValueError(f'Unable to stop execution. Reason: {response.text}')

    def download_result(self, execution_id: int, result_cls=Pipeline) -> Pipeline:
        response = requests.get(
            url=f'{self.controller_server}/executions/{self._current_project_id}/{execution_id}/download',
            cookies={USER_TOKEN_KEY: self._user_token},
            headers={GROUP_TOKEN_KEY: self.group_token},
            stream=True
        )

        if response.status_code != 200:
            raise ValueError(f'Unable to download results. Reason: {response.text}')

        tmp_path = f'_tmp_{int(datetime.utcnow().timestamp() * 1000)}'
        try:
            with open(tmp_path, 'wb') as tmp_file:
                shutil.copyfileobj(response.raw, tmp_file)
            shutil.unpack_archive(tmp_path, f'{self.output_path}/execution-{execution_id}', 'zip')
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass

        results_path_out = os.path.join(self.output_path,
                                        f'execution-{execution_id}',
                                        'out')
        results_folder = os.listdir(results_path_out)[0]
        load_path = os.path.join(results_path_out, results_folder, 'fitted_pipeline.json')
        pipeline = result_cls.from_serialized(load_path)

        clean_dir(results_path_out)
        return pipeline


def clean_dir(results_path_out):
    for root, dirs, files in os.walk(results_path_out):
        for file in files:
            os.remove(os.path.join(root, file))
