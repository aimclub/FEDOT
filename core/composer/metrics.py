from abc import abstractmethod

from sklearn.metrics import mean_squared_error, roc_auc_score

from core.composer.chain import Chain
from core.models.data import InputData


class ChainMetric:
    @staticmethod
    @abstractmethod
    def get_value(chain: Chain, reference_data: InputData) -> float:
        raise NotImplementedError()


class RmseMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData) -> float:
        results = chain.predict(reference_data)
        return mean_squared_error(y_true=reference_data.target, y_pred=results)


class MaeMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData) -> float:
        results = chain.predict(reference_data)
        return mean_squared_error(y_true=reference_data.target, y_pred=results)


class RocAucMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData) -> float:
        results = chain.predict(reference_data)
        try:
            # TODO re-factor to avoid negative
            score = -roc_auc_score(y_score=results.predict,
                                   y_true=reference_data.target)
            return score
        except Exception as ex:
            print(ex)
            return -0.5


# TODO: reference_data = None ?
class StructuralComplexityMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData) -> float:
        return chain.depth ** 2 + chain.length


class NodeNum(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData) -> float:
        return chain.length
