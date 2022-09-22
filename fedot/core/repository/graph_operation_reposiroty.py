from abc import ABC
from typing import List


class GraphOperationRepository(ABC):
    """ Base repository in order to extract suitable models for each of graph structure
    from specific files/configs etc """
    def __init__(self, **kwargs):
        pass

    def get_operations(self, **kwargs):
        """ Get models by specified model keys """
        raise NotImplementedError()

    def get_all_operations(self):
        """ Get all models with all keys """
        raise NotImplementedError()
