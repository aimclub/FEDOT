from abc import abstractmethod
from typing import Sequence, TypeVar

from fedot.core.utilities.serializable import Serializable

G = TypeVar('G', bound=Serializable)


class BaseRemoteEvaluator:
    """Interface for remote evaluator of graphs."""
    @property
    @abstractmethod
    def use_remote(self):
        return False

    @abstractmethod
    def compute_graphs(self, graphs: Sequence[G]) -> Sequence[G]:
        raise NotImplementedError()
