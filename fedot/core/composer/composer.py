from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import default_log
from fedot.core.optimisers.composer_requirements import ComposerRequirements
from fedot.core.optimisers.optimizer import GraphOptimizer
from fedot.core.pipelines.pipeline import Pipeline


class Composer(ABC):
    """
    Base class used for receiving composite operations via optimization

    :param optimiser: optimiser generated in :class:`~fedot.core.composer.ComposerBuilder`
    :param composer_requirements: requirements for composition process
    """

    def __init__(self, optimizer: GraphOptimizer, composer_requirements: Optional[ComposerRequirements] = None):
        self.composer_requirements = composer_requirements
        self.optimizer = optimizer
        self.log = default_log(self)

    @abstractmethod
    def compose_pipeline(self, data: Union[InputData, MultiModalData]) -> Union[Pipeline, Sequence[Pipeline]]:
        """
        Run composition process for optimal pipeline structure search

        :param data: Data used for problem solving

        :return: Best composed pipeline or pipelines.
            For single-objective optimization -- the best pipeline.
            For multi-objective optimization -- a list of the best pipelines is returned.
            Returned pipelines are ordered by the descending primary metric (the first is the best).
        """

        raise NotImplementedError()
