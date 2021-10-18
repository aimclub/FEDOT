import os
from typing import Optional

from fedot.core.log import default_log
from fedot.core.utils import default_fedot_data_dir


class Client:
    def __init__(self, connect_params: dict, exec_params: dict, output_path: Optional[str] = None):
        self.connect_params = connect_params
        self.exec_params = exec_params
        self.output_path = output_path if output_path else \
            os.path.join(default_fedot_data_dir(), 'remote_fit_results')
        self._logger = default_log('ClientLog')

    def create_task(self, config):
        raise NotImplementedError()

    def wait_until_ready(self):
        raise NotImplementedError()

    def download_result(self, execution_id):
        raise NotImplementedError()
