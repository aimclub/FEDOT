import os
import json
from fedot.core.repository.operation_types_repository import run_once


class DefaultOperationParamsRepository:

    __initialized_repository__ = {}

    def __init__(self, repository_name: str = 'default_operation_params.json'):
        repo_folder_path = str(os.path.dirname(__file__))
        file = os.path.join('data', repository_name)
        repo_path = os.path.join(repo_folder_path, file)
        DefaultOperationParamsRepository._initialise_repo(repo_path)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._repo_path = None

    @classmethod
    @run_once
    def _initialise_repo(cls, repo_path):
        with open(repo_path) as repository_json_file:
            repository_json = json.load(repository_json_file)
        cls.__initialized_repository__ = repository_json

    @classmethod
    def add_model_to_repository(cls, params: dict):
        cls.__initialized_repository__.update(params)

    @classmethod
    def get_default_params_for_operation(cls, model_name: str) -> dict:
        model_name = model_name.split('/')[0]
        if model_name in cls.__initialized_repository__:
            return cls.__initialized_repository__[model_name]
        return {}

