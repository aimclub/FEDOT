from abc import ABC
from typing import List


class GraphOperationRepository(ABC):
    """ Base repository in order to extract suitable models for each of graph structure
    from specific files/configs etc
    :param models_keys: possible keys """
    def __init__(self, models_keys: List[str] = None, **kwargs):
        self.models_keys = models_keys

    def get_operations(self, **kwargs):
        """ Get models by specified model keys """
        raise NotImplementedError()

    def get_all_operations(self):
        """ Get all models with all keys """
        raise NotImplementedError()
