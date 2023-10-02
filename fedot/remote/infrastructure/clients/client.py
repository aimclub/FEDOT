import os
from abc import abstractmethod
from datetime import timedelta
from typing import Optional, Type, TypeVar

from golem.core.log import default_log
from golem.core.utilities.serializable import Serializable

from fedot.core.utils import default_fedot_data_dir
from fedot.utilities.custom_errors import AbstractMethodNotImplementError

G = TypeVar('G', bound=Serializable)


class Client:
    """
    Base class for remote evaluation client. It allows to fit the pipelines in external system instead local one.
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
        self._logger = default_log(prefix='ClientLog')

    @abstractmethod
    def create_task(self, config: dict):
        """
        Create task for execution
        :param config - configuration of pipeline fitting
        :return: id of created task
        """
        raise AbstractMethodNotImplementError

    @abstractmethod
    def wait_until_ready(self) -> timedelta:
        """
        Delay execution until all remote tasks are ready
        :return: waiting time
        """
        raise AbstractMethodNotImplementError

    @abstractmethod
    def download_result(self, execution_id: int, result_cls: Type[G]) -> G:
        """
        :param execution_id: id of remote task
        :param result_cls: result
        :return: fitted pipeline downloaded from the remote server
        """
        raise AbstractMethodNotImplementError
