from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from fedot.core.log import default_log
from fedot.core.visualisation.opt_history.arg_constraint_wrapper import ArgConstraintWrapper

if TYPE_CHECKING:
    from fedot.core.optimisers.opt_history import OptHistory


class HistoryVisualization(metaclass=ArgConstraintWrapper):
    constraint_checkers = []  # Use this for class-specific constraint checkers.

    def __init__(self, history: OptHistory):
        self.log = default_log(self)
        self.history = history

    @abstractmethod
    def visualize(self):
        raise NotImplementedError
