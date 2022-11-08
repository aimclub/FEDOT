from numbers import Real
from typing import Callable, Union, TypeVar

from fedot.core.composer.metrics import (ComputationTime, Accuracy, F1, Logloss, MAE,
                                         MAPE, SMAPE, MSE, MSLE, Metric, NodeNum, Precision, R2,
                                         RMSE, ROCAUC, Silhouette, StructuralComplexity)
from fedot.core.dag.graph import Graph
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class MetricsEnum(Enum):
    pass


G = TypeVar('G', bound=Graph, covariant=True)
MetricCallable = Callable[[G], Real]
MetricType = Union[MetricCallable, MetricsEnum]


class QualityMetricsEnum(MetricsEnum):
    pass


class ComplexityMetricsEnum(MetricsEnum):
    node_num = 'node_number'
    structural = 'structural'
    computation_time = 'computation_time_in_seconds'


class ClusteringMetricsEnum(QualityMetricsEnum):
    silhouette = 'silhouette'


class ClassificationMetricsEnum(QualityMetricsEnum):
    ROCAUC = 'roc_auc'
    precision = 'precision'
    f1 = 'f1'
    logloss = 'neg_log_loss'
    ROCAUC_penalty = 'roc_auc_pen'
    accuracy = 'accuracy'


class RegressionMetricsEnum(QualityMetricsEnum):
    RMSE = 'rmse'
    MSE = 'mse'
    MSLE = 'neg_mean_squared_log_error'
    MAPE = 'mape'
    SMAPE = 'smape'
    MAE = 'mae'
    R2 = 'r2'
    RMSE_penalty = 'rmse_pen'


class MetricsRepository:
    _metrics_implementations = {
        # classification
        ClassificationMetricsEnum.ROCAUC: ROCAUC.get_value,
        ClassificationMetricsEnum.ROCAUC_penalty: ROCAUC.get_value_with_penalty,
        ClassificationMetricsEnum.f1: F1.get_value,
        ClassificationMetricsEnum.precision: Precision.get_value,
        ClassificationMetricsEnum.accuracy: Accuracy.get_value,
        ClassificationMetricsEnum.logloss: Logloss.get_value,

        # regression
        RegressionMetricsEnum.MAE: MAE.get_value,
        RegressionMetricsEnum.MSE: MSE.get_value,
        RegressionMetricsEnum.MSLE: MSLE.get_value,
        RegressionMetricsEnum.MAPE: MAPE.get_value,
        RegressionMetricsEnum.SMAPE: SMAPE.get_value,
        RegressionMetricsEnum.RMSE: RMSE.get_value,
        RegressionMetricsEnum.RMSE_penalty: RMSE.get_value_with_penalty,
        RegressionMetricsEnum.R2: R2.get_value,

        # clustering
        ClusteringMetricsEnum.silhouette: Silhouette.get_value,

        # structural
        ComplexityMetricsEnum.structural: StructuralComplexity.get_value,
        ComplexityMetricsEnum.node_num: NodeNum.get_value,
        ComplexityMetricsEnum.computation_time: ComputationTime.get_value
    }

    @staticmethod
    def metric_by_id(metric_id: MetricsEnum, default_callable: MetricCallable = None) -> MetricCallable:
        return MetricsRepository._metrics_implementations.get(metric_id, default_callable)

    @staticmethod
    def metric_class_by_id(metric_id: MetricsEnum) -> Metric:
        return MetricsRepository._metrics_implementations[metric_id].__self__()
