from dataclasses import dataclass
from typing import Callable, Union

from fedot.core.optimisers.opt_history import OptHistory


@dataclass
class TimeoutParams:
    test_input: dict
    test_answer: Union[Callable[[OptHistory], bool], BaseException]
