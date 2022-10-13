from typing import Optional, List, Callable, Tuple, Iterable

from fedot.core.log import default_log

ConditionType = Callable[[], bool]
ConditionEntryType = Tuple[ConditionType, Optional[str]]


class GroupedCondition:
    """Represents sequence of ordinary conditions with logging.
    All composed conditions are combined with reduce function on booleans.

    By the default 'any' is used, so in this case the grouped condition is True
    if any of the composed conditions is True. The message corresponding
    to the actual fired condition is logged (if it was provided)."""

    def __init__(self, conditions_reduce: Callable[[Iterable[bool]], bool] = any, results_as_message: bool = False):
        self._reduce = conditions_reduce
        self._conditions: List[ConditionEntryType] = []
        self._log = default_log(self)
        self._results_as_message = results_as_message

    def add_condition(self, condition: ConditionType, log_msg: Optional[str] = None) -> 'GroupedCondition':
        """Builder-like method for adding conditions."""
        self._conditions.append((condition, log_msg))
        return self

    def __bool__(self):
        return self()

    def __call__(self) -> bool:
        return self._reduce(map(self._check_condition, self._conditions))

    def _check_condition(self, entry: ConditionEntryType) -> bool:
        cond, msg = entry
        res = cond()
        if res and msg:
            if self._results_as_message:
                self._log.message(msg)
            else:
                self._log.info(msg)
        return res
