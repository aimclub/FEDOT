from random import choice
from typing import Optional, List

from fedot.core.pipelines.pipeline import Pipeline


class OperationsFilter:
    def satisfies(self, pipeline: Optional[Pipeline]) -> bool:
        """ Checks if all operations in a Pipeline satisify this filter. """
        return True

    def sample(self) -> str:
        """ Samples some operation that satisfies this filter. """
        raise NotImplementedError()


class WhitelistOperationsFilter(OperationsFilter):
    """ Simple OperationsFilter implementation based on two lists:
    one for all available operations, another for sampling operations. """

    def __init__(self, available_ops: List[str], available_task_ops: Optional[List[str]] = None):
        super().__init__()
        self._whitelist = tuple(available_ops)
        self._choice_ops = tuple(available_task_ops) if available_task_ops else self._whitelist

    def satisfies(self, pipeline: Optional[Pipeline]) -> bool:
        def node_ok(node):
            return node.operation.operation_type in self._whitelist

        return pipeline and all(map(node_ok, pipeline.nodes))

    def sample(self) -> str:
        return choice(self._choice_ops)
