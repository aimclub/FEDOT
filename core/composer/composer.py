from abc import ABC, abstractmethod
from typing import (
    List,
    Any,
    Callable
)

from core.composer.chain import Chain


# TODO: specify ComposerRequirements class
class Composer(ABC):
    @abstractmethod
    def compose_chain(self, initial_chain: Chain, requirements: List[Any],
                      metrics: Callable):
        raise NotImplementedError()
