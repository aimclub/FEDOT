from abc import abstractmethod

from fedot.core.data.data import InputData


class Explainer:
    """
    This class is an abstract class for various explanation methods.
    """
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def output(self, *args, **kwargs):
        raise NotImplementedError
