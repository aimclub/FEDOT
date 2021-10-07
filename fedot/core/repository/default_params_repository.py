import os
import json


class DefaultOperationParamsRepository:
    def __init__(self, repository_name: str = 'default_operation_params.json'):
        repo_folder_path = str(os.path.dirname(__file__))
        file = os.path.join('data', repository_name)
        self._repo_path = os.path.join(repo_folder_path, file)
        self._repo = self._initialise_repo()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._repo_path = None

    def _initialise_repo(self) -> dict:
        with open(self._repo_path) as repository_json_file:
            repository_json = json.load(repository_json_file)

        return repository_json

    def get_default_params_for_operation(self, model_name: str) -> dict:
        model_name = model_name.split('/')[0]
        if model_name in self._repo:
            return self._repo[model_name]
        return {}
