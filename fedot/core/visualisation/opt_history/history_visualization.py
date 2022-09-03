from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from fedot.core.log import default_log
from fedot.core.visualisation.opt_history.arg_constraint_wrapper import ArgConstraintWrapper

if TYPE_CHECKING:
    from fedot.core.optimisers.opt_history import OptHistory


class HistoryVisualization(metaclass=ArgConstraintWrapper):
    """ Base class for creating visualizations of FEDOT optimization history.
    The only necessary method is 'visualize' - it must show or save the plot in any form after the call.

    One should refer the OptHistory instance as `self.history` to be able to connect one's visualization
    to `OptHistory.show()`. See the examples of connecting visualizations in the module `opt_viz.py`.

    It is good practice to be aware of constraints on your visualizations. You can either implement
    default constraints that will catch your kwarg across all the visualizations or define your single
    class specific constraints by assigning them to `constraint_checkers` class attribute.
    See `fedot.core.visualisation.opt_history.arg_constraint_wrapper.py` for examples.
    """
    constraint_checkers = []  # Use this for class-specific constraint checkers.

    def __init__(self, history: OptHistory):
        self.log = default_log(self)
        self.history = history

    @abstractmethod
    def visualize(self, *args, **kwargs):
        raise NotImplementedError()
