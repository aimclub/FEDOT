import dataclasses
from dataclasses import dataclass
from numbers import Number
from typing import Sequence

from fedot.core.optimisers.composer_requirements import ComposerRequirements


@dataclass
class PipelineComposerRequirements(ComposerRequirements):
    """Defines restrictions and requirements for composition of final graphs.

    Restrictions on final graphs:
    :param start_depth: start value of adaptive tree depth
    :param max_depth: max depth of the resulting pipeline
    :param min_arity: min number of parents for node
    :param max_arity: max number of parents for node

    :param primary: operation types for :class:`~fedot.core.pipelines.node.PrimaryNode`s
    :param secondary: operation types for :class:`~fedot.core.pipelines.node.SecondaryNode`s

    """

    start_depth: int = 3
    max_depth: int = 3
    min_arity: int = 2
    max_arity: int = 2

    primary: Sequence[str] = tuple()
    secondary: Sequence[str] = tuple()

    def __post_init__(self):
        super().__post_init__()
        excluded_fields = ['n_jobs']
        for field_name, field_value in dataclasses.asdict(self).items():
            if field_name in excluded_fields:
                continue
            if isinstance(field_value, Number) and field_value < 0:
                raise ValueError(f'Value of {field_name} must be non-negative')
