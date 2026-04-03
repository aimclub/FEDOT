"""
Interfaces for trainers and hookable objects
"""

from abc import ABC, abstractmethod
from typing import Protocol, Iterable, Any, Dict, Optional, runtime_checkable
from enum import Enum


@runtime_checkable
class IHookable(Protocol):
    """Interface for objects that can register and manage hooks"""

    def register_additional_hooks(self, hooks: Iterable[Enum]) -> None:
        """Register additional hooks for training"""
        ...

    def _init_hooks(self) -> None:
        """Initialize hooks for the model"""
        ...


class ITrainer(ABC):
    """Interface for training functionality"""

    @abstractmethod
    def fit(self, input_data: Any, supplementary_data: Optional[Dict] = None, loader_type: str = 'train') -> Any:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, input_data: Any, output_mode: str = "default") -> Any:
        """Make predictions"""
        pass

    # @abstractmethod
    # def save_model(self, path: str) -> None:
    #     """Save the model"""
    #     pass

    # @abstractmethod
    # def load_model(self, path: str) -> None:
    #     """Load the model"""
    #     pass 