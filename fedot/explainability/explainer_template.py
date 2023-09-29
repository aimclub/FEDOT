from abc import abstractmethod

from fedot.utilities.custom_errors import AbstractMethodNotImplementError


class Explainer:
    """
    An abstract class for various explanation methods.
    """

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def explain(self, *args, **kwargs):
        raise AbstractMethodNotImplementError

    @abstractmethod
    def visualize(self, *args, **kwargs):
        raise AbstractMethodNotImplementError
