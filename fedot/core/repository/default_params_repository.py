import json
import os
from copy import deepcopy
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
        # Opt-in, process-scoped experiment profile, inherited by worker processes.
        # Never mutate the cached built-in repository or overwrite explicit node parameters.
        self._overlay = {}
        overlay_path = os.environ.get('FEDOT_OPERATION_DEFAULTS_OVERLAY')
        if overlay_path and repository_name == 'default_operation_params.json':
            if not os.path.isabs(overlay_path):
                raise ValueError('FEDOT_OPERATION_DEFAULTS_OVERLAY must be an absolute path')
            overlay = _load_repository(overlay_path)
            if not isinstance(overlay, dict) or any(
                    not isinstance(name, str) or not isinstance(params, dict)
                    for name, params in overlay.items()):
                raise ValueError('Operation defaults overlay must map operation names to parameter dictionaries')
            self._overlay = overlay

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._repo_path = None

    def _initialise_repo(self) -> dict:
        return _load_repository(self._repo_path)

    def get_default_params_for_operation(self, model_name: str) -> dict:
        model_name = model_name.split('/')[0]
        return deepcopy({**self._repo.get(model_name, {}), **self._overlay.get(model_name, {})})
