import json
import os
from functools import lru_cache


@lru_cache(maxsize=None)
def _load_repository(repo_path: str) -> dict:
    """Parse the repository file once per process.

    Default parameters are requested in ``PipelineNode.__init__``, so every
    graph the optimiser mutates, verifies, restores or serialises used to
    re-read the JSON from disk for each of its nodes - thousands of reads per
    generation, a noticeable share of the time between two populations.
    """
    with open(repo_path) as repository_json_file:
        return json.load(repository_json_file)


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
        return _load_repository(self._repo_path)

    def get_default_params_for_operation(self, model_name: str) -> dict:
        model_name = model_name.split('/')[0]
        return self._repo.get(model_name, {})
