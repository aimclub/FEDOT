from typing import Sequence, Optional, Union, Callable

from fedot.core.dag.graph import Graph
from fedot.core.log import default_log
from fedot.core.optimisers.adapters import BaseOptimizationAdapter, DirectAdapter
from fedot.core.optimisers.graph import OptGraph

# Validation rule can either return False or raise a ValueError to signal a failed check
VerifierRuleType = Callable[..., bool]


class GraphVerifier:
    def __init__(self,
                 rules: Sequence[VerifierRuleType] = (),
                 adapter: Optional[BaseOptimizationAdapter] = None):
        self._rules = rules
        self._adapter = adapter or DirectAdapter()
        self._log = default_log(self)

    def __call__(self, graph: Union[Graph, OptGraph]) -> bool:
        return self.verify(graph)

    def verify(self, graph: Union[Graph, OptGraph]) -> bool:
        restored_graph: Graph = self._adapter.restore(graph)
        # Check if all rules pass
        for rule in self._rules:
            try:
                if rule(restored_graph) is False:
                    return False
            except ValueError as err:
                self._log.debug(f'Graph verification failed with error <{err}> '
                                f'for rule={rule} on graph={restored_graph.root_node.descriptive_id}.')
                return False
        return True
