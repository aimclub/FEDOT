from abc import abstractmethod

from sklearn.metrics import mean_squared_error, roc_auc_score

from core.chain_validation import validate
from core.composer.chain import Chain
from core.models.data import InputData
from experiments.tree_dist import chain_distance


def from_maximised_metric(metric_func):
    def wrapper(*args, **kwargs):
        return -metric_func(*args, **kwargs)

    return wrapper


class ChainMetric:
    @staticmethod
    @abstractmethod
    def get_value(chain: Chain, reference_data: InputData, **kwargs) -> float:
        raise NotImplementedError()


class RmseMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData, **kwargs) -> float:
        results = chain.predict(reference_data)
        return mean_squared_error(y_true=reference_data.target, y_pred=results.predict)


class MaeMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData, **kwargs) -> float:
        results = chain.predict(reference_data)
        return mean_squared_error(y_true=reference_data.target, y_pred=results.predict)


class RocAucMetric(ChainMetric):
    @staticmethod
    @from_maximised_metric
    def get_value(chain: Chain, reference_data: InputData, **kwargs) -> float:
        try:
            validate(chain)
            results = chain.predict(reference_data)
            score = round(roc_auc_score(y_score=results.predict,
                                        y_true=reference_data.target), 3)
        except Exception as ex:
            print(ex)
            score = 0.5

        return score


# TODO: reference_data = None ?
class StructuralComplexityMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData, **kwargs) -> float:
        return chain.depth ** 2 + chain.length


class NodeNum(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData, **kwargs) -> float:
        return chain.length


class ChainDistanceMetric(ChainMetric):
    @staticmethod
    def get_value(chain: Chain, reference_data: InputData, **kwargs) -> float:
        if 'source_chain' in kwargs:
            source = kwargs['source_chain']
            distance = chain_distance(chain, source)
            return distance
