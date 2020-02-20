from abc import abstractmethod

from sklearn.metrics import mean_squared_error, roc_auc_score

from core.composer.chain import Chain
from core.models.data import Data


class ChainMetric:
    @staticmethod
    @abstractmethod
    def get_value(chain: Chain, reference_data: Data) -> float:
        raise NotImplemented


class RmseMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: Data) -> float:
        results = chain.evaluate()
        return mean_squared_error(y_true=reference_data.target, y_pred=results.target)


class MaeMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: Data) -> float:
        results = chain.evaluate()
        return mean_squared_error(y_true=reference_data.target, y_pred=results.target)


class RocAucMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: Data) -> float:
        results = chain.evaluate()
        return roc_auc_score(y_score=results.target, y_true=reference_data)


class StructuralComplexityMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: Data) -> float:
        return chain.depth ** 2 + chain.length


class NodeNum(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: Data) -> float:
        return chain.length
