import datetime
import json
import os
import shutil
import time
from typing import List

import requests

USER_TOKEN_KEY = "x-jwt-auth"
GROUP_TOKEN_KEY = "x-jwt-models-execution"


class Client:

    def __init__(self, authorization_server: str = None, controller_server: str = None):

        self.authorization_server = "http://10.32.0.51:30880/b" if authorization_server is None else authorization_server
        self.controller_server = "http://10.32.0.51:30880/models-controller" if controller_server is None else controller_server
        self._user_token = None
        self.user = None
        self._current_project_id = None
        self.group_token = None

    def login(self, login: str, password: str) -> None:
        response = requests.post(
            url=f"{self.authorization_server}/users/login",
            json={
                "login": login,
                "password": password
            }
        )

        if response.status_code != 200:
            raise ValueError(f"Unable to get user token! Reason: {response.text}")

        self._user_token = response.cookies["x-jwt-auth"]
        self.user = json.loads(response.text)

    def set_user_token(self, token):
        self._user_token = token

    @property
    def user_token(self):
        return self._user_token

    def set_group_token(self, project_id: int, group_token: str = None, group_id: int = None) -> None:
        if group_token is not None:
            self.group_token = group_token
            self._current_project_id = project_id
            return

        if group_id is not None:
            group = self.get_execution_group(
                project_id=project_id,
                group_id=group_id
            )

            self.group_token = group["token"]
            self._current_project_id = group["project_id"]

            return

        raise ValueError(f"You have to specify project_id and token/group_id!")

    def get_execution_groups(self, project_id: int) -> List[dict]:
        response = requests.get(
            url=f"{self.controller_server}/execution-groups/{project_id}",
            cookies={USER_TOKEN_KEY: self._user_token}
        )

        if response.status_code != 200:
            raise ValueError(f"Unable to get execution groups. Reason: {response.text}")

        return json.loads(response.text)

    def get_execution_group(self, project_id: int, group_id: int) -> dict:
        response = requests.get(
            url=f"{self.controller_server}/execution-groups/{project_id}/{group_id}",
            cookies={USER_TOKEN_KEY: self._user_token}
        )

        if response.status_code != 200:
            raise ValueError(f"Unable to get execution group. Reason: {response.text}")

        return json.loads(response.text)

    def delete_execution_group(self, project_id: int, group_id: int) -> None:
        raise NotImplementedError

        response = requests.delete(
            url=f"{self.controller_server}/execution-groups/{project_id}/{group_id}",
            cookies={USER_TOKEN_KEY: self._user_token}
        )

        if response.status_code != 204:
            raise ValueError(f"Unable to delete execution group. Reason: {response.text}")

    def create_execution_group(self, project_id: int) -> dict:
        response = requests.post(
            url=f"{self.controller_server}/execution-groups/{project_id}",
            cookies={USER_TOKEN_KEY: self._user_token}
        )

        if response.status_code != 200:
            raise ValueError(f"Unable to create execution group. Reason: {response.text}")

        return json.loads(response.text)

    def _get_executions(self):
        response = requests.get(
            url=f"{self.controller_server}/executions/{self._current_project_id}",
            cookies={USER_TOKEN_KEY: self._user_token},
            headers={GROUP_TOKEN_KEY: self.group_token}
        )

        if response.status_code != 200:
            raise ValueError(f"Unable to get executions. Reason: {response.text}")

        return json.loads(response.text)

    def get_executions(self, wait_until_finished: bool = False):
        if not wait_until_finished:
            return self._get_executions()

        while True:
            executions = self._get_executions()
            if len([ex for ex in executions if ex["status"] in ["Created", "Waiting", "Pending", "Running"]]) == 0:
                return executions
            time.sleep(2)

    def get_execution(self, execution_id: int):
        response = requests.get(
            url=f"{self.controller_server}/executions/{self._current_project_id}/{execution_id}",
            cookies={USER_TOKEN_KEY: self._user_token},
            headers={GROUP_TOKEN_KEY: self.group_token}
        )

        if response.status_code != 200:
            raise ValueError(f"Unable to get execution. Reason: {response.text}")

        return json.loads(response.text)

    def create_execution(self, container_input_path: str,
                         container_output_path: str,
                         container_config_path: str,
                         container_image: str,
                         timeout: int,
                         config: bytes) -> dict:
        response = requests.post(
            url=f"{self.controller_server}/executions/{self._current_project_id}",
            cookies={USER_TOKEN_KEY: self._user_token},
            headers={GROUP_TOKEN_KEY: self.group_token},
            files={
                "input_path": (None, container_input_path),
                "output_path": (None, container_output_path),
                "config_path": (None, container_config_path),
                "image": (None, container_image),
                "timeout": (None, timeout),
                "config_file": ("config", config)
            }
        )

        if response.status_code != 200:
            raise ValueError(f"Unable to create execution. Reason: {response.text}")

        return json.loads(response.text)

    def stop_execution(self, execution_id: int) -> None:
        response = requests.delete(
            url=f"{self.controller_server}/executions/{self._current_project_id}/{execution_id}",
            cookies={USER_TOKEN_KEY: self._user_token},
            headers={GROUP_TOKEN_KEY: self.group_token}
        )

        if response.status_code != 204:
            raise ValueError(f"Unable to stop execution. Reason: {response.text}")

        return

    def download_result(self, execution_id: int, path: str, unpack: bool = False) -> None:
        response = requests.get(
            url=f"{self.controller_server}/executions/{self._current_project_id}/{execution_id}/download",
            cookies={USER_TOKEN_KEY: self._user_token},
            headers={GROUP_TOKEN_KEY: self.group_token},
            stream=True
        )

        if response.status_code != 200:
            raise ValueError(f"Unable to download results. Reason: {response.text}")

        if not unpack:
            with open(path, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            return
        else:
            tmp_path = f"_tmp_{int(datetime.datetime.utcnow().timestamp() * 1000)}"
            try:
                with open(tmp_path, "wb") as tmp_file:
                    shutil.copyfileobj(response.raw, tmp_file)
                shutil.unpack_archive(tmp_path, f"{path}/execution-{execution_id}", "zip")
            except Exception as e:
                raise e
            finally:
                try:
                    os.remove(tmp_path)
                except FileNotFoundError:
                    pass
