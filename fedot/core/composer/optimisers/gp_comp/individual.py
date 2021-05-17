from dataclasses import dataclass, field
from typing import List

from fedot.core.chains.chain import Chain
from fedot.core.composer.composing_history import ParentOperator


@dataclass
class Individual:
    chain: Chain  # TODO replace to Graph after merge
    fitness: List[float] = None
    parent_operators: List[ParentOperator] = field(default_factory=lambda: [])
