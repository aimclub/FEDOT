from copy import copy
from typing import Optional, Dict, Any

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.optimisers.graph import OptGraph

from test.unit.dag.test_graph_utils import nodes_same


class MockDomainStructure:
    """Mock domain structure for testing adapt/restore logic.
    Represents just a list of nodes."""

    def __init__(self, nodes):
        self.nodes = copy(nodes)

    def __eq__(self, other):
        return nodes_same(self.nodes, other.nodes)


class MockAdapter(BaseOptimizationAdapter[MockDomainStructure]):
    def __init__(self):
        super().__init__(base_graph_class=MockDomainStructure)

    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> MockDomainStructure:
        return MockDomainStructure(opt_graph.nodes)

    def _adapt(self, adaptee: MockDomainStructure) -> OptGraph:
        return OptGraph(adaptee.nodes)
