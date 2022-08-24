from abc import abstractmethod

from fedot.core.log import default_log


class HistoryVisualization:

    def __init__(self, history):
        self.log = default_log(self)
        self.history = history

    @abstractmethod
    def visualize(self):
        raise NotImplementedError
