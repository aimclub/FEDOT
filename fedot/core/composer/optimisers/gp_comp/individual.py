import gc
from typing import List

from fedot.core.chains.chain import Chain
from fedot.core.composer.composing_history import ParentOperator


class Individual:
    def __init__(self, chain: Chain, fitness: List[float] = None,
                 parent_operators: List[ParentOperator] = None):
        self.parent_operators = parent_operators if parent_operators is not None else []
        self.fitness = fitness

        self.chain = _release_chain_resources(chain)


def _release_chain_resources(chain: Chain):
    """
    Remove all 'heavy' parts from the chain
    :param chain: fitted chain
    :return: chain without fitted models and data
    """
    chain.unfit()
    gc.collect()
    return chain
