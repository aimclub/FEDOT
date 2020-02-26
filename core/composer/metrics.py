from abc import abstractmethod

from sklearn.metrics import mean_squared_error, roc_auc_score

from core.composer.chain import Chain


class ChainMetric:
    @staticmethod
    @abstractmethod
    def get_value(chain: Chain) -> float:
        raise NotImplementedError()


class RmseMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain) -> float:
        results = chain.evaluate()
        return mean_squared_error(y_true=chain.reference_data.target, y_pred=results)


class MaeMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain) -> float:
        results = chain.evaluate()
        return mean_squared_error(y_true=chain.reference_data.target, y_pred=results)


class RocAucMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain) -> float:
        results = chain.evaluate()
        return roc_auc_score(y_score=results.predict, y_true=chain.reference_data.target)


class StructuralComplexityMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain) -> float:
        return chain.depth ** 2 + chain.length


class NodeNum(ChainMetric):
    @staticmethod
    def get_value(chain: Chain) -> float:
        return chain.length
