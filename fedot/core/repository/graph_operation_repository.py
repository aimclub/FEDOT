from abc import ABC, abstractmethod


class GraphOperationRepository(ABC):
    """ Base repository in order to extract suitable models for each of graph structure
    from specific files/configs etc """

    @abstractmethod
    def get_operations(self, **kwargs):
        """ Get models by specified model keys """
        raise NotImplementedError()

    @abstractmethod
    def get_all_operations(self):
        """ Get all models with all keys """
        raise NotImplementedError()
