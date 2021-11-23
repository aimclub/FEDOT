import os
from typing import Optional

from fedot.core.log import default_log
from fedot.core.utils import default_fedot_data_dir


class Client:
    """
    Base class for remote evaluation client. It allow fitting the pipelines in external system instead local one.
    """

    def __init__(self, connect_params: dict, exec_params: dict, output_path: Optional[str] = None):
        """
        :param connect_params: parameters for connection to remote server
        :param exec_params: params for remote task execution
        :param output_path: local path for temporary saving of downloaded pipelines
        """
        self.connect_params = connect_params
        self.exec_params = exec_params
        self.output_path = output_path if output_path else \
            os.path.join(default_fedot_data_dir(), 'remote_fit_results')
        self._logger = default_log('ClientLog')

    def create_task(self, config):
        """
        Create task for execution
        :param config - configuration of pipeline fitting
        :return: id of created task
        """
        raise NotImplementedError()

    def wait_until_ready(self) -> float:
        """
        Delay execution until all remote tasks are ready
        :return: waiting time
        """
        raise NotImplementedError()

    def download_result(self, execution_id):
        """
        :param execution_id: id of remote task
        :return: fitted pipeline downloaded from the remote server
        """
        raise NotImplementedError()
