from random import randint

from core.composer.chain import Chain
from core.composer.metrics import ChainMetric


class RandomMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain) -> float:
        return randint(0, 1000)
