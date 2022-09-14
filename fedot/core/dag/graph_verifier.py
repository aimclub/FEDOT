from typing import Sequence, Optional, Callable

from fedot.core.adapter import BaseOptimizationAdapter
from fedot.core.adapter.adapter import IdentityAdapter
from fedot.core.dag.graph import Graph
from fedot.core.log import default_log

# Validation rule can either return False or raise a ValueError to signal a failed check
VerifierRuleType = Callable[..., bool]


class VerificationError(ValueError):
    pass


class GraphVerifier:
    def __init__(self, rules: Sequence[VerifierRuleType] = (),
                 adapter: Optional[BaseOptimizationAdapter] = None,
                 raise_on_failure: bool = False):
        self._adapter = adapter or IdentityAdapter()
        self._rules = rules
        self._log = default_log(self)
        self._raise = raise_on_failure

    def __call__(self, graph: Graph) -> bool:
        return self.verify(graph)

    def verify(self, graph: Graph) -> bool:
        # Check if all rules pass
        adapt = self._adapter.adapt_func
        for rule in self._rules:
            try:
                if adapt(rule)(graph) is False:
                    return False
            except ValueError as err:
                msg = f'Graph verification failed with error <{err}> '\
                      f'for rule={rule} on graph={graph.descriptive_id}.'
                if self._raise:
                    raise VerificationError(msg)
                else:
                    self._log.debug(msg)
                    return False
        return True
