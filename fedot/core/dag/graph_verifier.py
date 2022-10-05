from typing import Sequence, Optional, Callable

from fedot.core.adapter import BaseOptimizationAdapter, DirectAdapter
from fedot.core.dag.graph import Graph
from fedot.core.log import default_log

# Validation rule can either return False or raise a ValueError to signal a failed check
VerifierRuleType = Callable[..., bool]


class GraphVerifier:
    def __init__(self, rules: Sequence[VerifierRuleType] = (),
                 adapter: Optional[BaseOptimizationAdapter] = None):
        self._adapter = adapter or DirectAdapter()
        self._rules = rules
        self._log = default_log(self)

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
                self._log.debug(f'Graph verification failed with error <{err}> '
                                f'for rule={rule} on graph={graph.descriptive_id}.')
                return False
        return True
