from abc import abstractmethod
from inspect import stack


class Explainer:
    """
    An abstract class for various explanation methods.
    """
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def explain(self, *args, **kwargs):
        raise NotImplementedError(f'Method {stack()[0][3]} not implemented in {self.__class__}')

    @abstractmethod
    def visualize(self, *args, **kwargs):
        raise NotImplementedError(f'Method {stack()[0][3]} not implemented in {self.__class__}')
