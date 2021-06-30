import gc
from typing import List

from fedot.core.chains.chain import Chain
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import ParentOperator

ERROR_PREFIX = 'Invalid graph configuration:'


class Individual:
    def __init__(self, graph: 'OptGraph', fitness: List[float] = None,
                 parent_operators: List[ParentOperator] = None):
        self.parent_operators = parent_operators if parent_operators is not None else []
        self.fitness = fitness

        # TODO remove direct chain reference
        self.graph = _release_chain_resources(graph) if isinstance(graph, Chain) else graph


def _release_chain_resources(chain: Chain):
    """
    Remove all 'heavy' parts from the graph
    :param chain: fitted chain
    :return: chain without fitted models and data
    """
    chain.unfit()
    gc.collect()
    return chain
