import os
from datetime import timedelta
from typing import Optional, Callable
from uuid import uuid4

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import default_fedot_data_dir
from fedot.remote.infrastructure.clients.client import Client
from fedot.remote.run_pipeline import fit_pipeline


class TestClient(Client):
    def __init__(self, connect_params: dict, exec_params: dict, output_path: Optional[str] = None):
        self.connect_params = connect_params
        self.exec_params = exec_params
        self.output_path = output_path if output_path else \
            os.path.join(default_fedot_data_dir(), 'remote_fit_results')
        super().__init__(connect_params, exec_params, output_path)

    def create_task(self, config) -> str:
        fit_pipeline(config)
        return str(uuid4())

    def wait_until_ready(self) -> timedelta:
        return timedelta()

    def download_result(self, execution_id: int, result_cls=Pipeline) -> Pipeline:
        results_path_out = os.path.join(self.output_path)
        results_folder = os.listdir(results_path_out)[0]
        pipeline = result_cls.from_serialized(os.path.join(results_path_out, results_folder, 'fitted_pipeline.json'))
        return pipeline
