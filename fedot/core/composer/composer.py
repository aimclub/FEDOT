from abc import ABC, abstractmethod
from typing import List, Optional, Union

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import default_log
from fedot.core.optimisers.composer_requirements import ComposerRequirements
from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory
from fedot.core.optimisers.optimizer import GraphOptimizer
from fedot.core.pipelines.pipeline import Pipeline


class Composer(ABC):
    """
    Base class used for receiving composite operations via optimization

    Args:
        optimizer: optimizer generated in :class:`~fedot.core.composer.ComposerBuilder`
        composer_requirements: requirements for composition process
    """

    def __init__(self, optimizer: GraphOptimizer, composer_requirements: Optional[ComposerRequirements] = None):
        self.composer_requirements = composer_requirements
        self.optimizer = optimizer
        self.log = default_log(self)

    @property
    def history(self) -> OptHistory:
        return self.optimizer.history

    @abstractmethod
    def compose_pipeline(self, data: Union[InputData, MultiModalData]) -> Union[Pipeline, List[Pipeline]]:
        """
        Run composition process for optimal pipeline structure search

        Args:
            data: Data used for problem solving

        Returns:
            For ``single-objective optimization`` -> the best pipeline.\n
            For ``multi-objective optimization`` -> a list of the best pipelines is returned.

        Notes:
            Returned pipelines are ordered by the descending primary metric (the first is the best).
        """

        raise NotImplementedError()
