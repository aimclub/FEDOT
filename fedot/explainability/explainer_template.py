from abc import abstractmethod


class Explainer:
    """
    An abstract class for various explanation methods.
    """
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def explain(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def visualize(self, *args, **kwargs):
        raise NotImplementedError
