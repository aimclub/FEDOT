from typing import Sequence, Optional, Callable

from fedot.core.dag.graph import Graph
from fedot.core.log import default_log
from fedot.core.optimisers.adapt_registry import restore

# Validation rule can either return False or raise a ValueError to signal a failed check
VerifierRuleType = Callable[..., bool]


class GraphVerifier:
    def __init__(self, rules: Sequence[VerifierRuleType] = ()):
        self._rules = rules
        self._log = default_log(self)

    def __call__(self, graph: Graph) -> bool:
        return self.verify(graph)

    def verify(self, graph: Graph) -> bool:
        # Check if all rules pass
        for rule in self._rules:
            try:
                if restore(rule)(graph) is False:
                    return False
            except ValueError as err:
                self._log.debug(f'Graph verification failed with error <{err}> '
                                f'for rule={rule} on graph={graph.root_node.descriptive_id}.')
                return False
        return True
