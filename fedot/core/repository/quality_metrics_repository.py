from typing import Callable

from fedot.core.composer.metrics import (ComputationTime, Accuracy, F1, Logloss, MAE,
                                         MAPE, MSE, MSLE, Metric, NodeNum, Precision, R2,
                                         RMSE, ROCAUC, Silhouette, StructuralComplexity)
from fedot.core.utils import ComparableEnum as Enum


class MetricsEnum(Enum):
    pass


class QualityMetricsEnum(MetricsEnum):
    pass


class ComplexityMetricsEnum(MetricsEnum):
    node_num = 'node_number'
    structural = 'structural'
    computation_time = 'computation_time'


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

    def metric_by_id(self, metric_id: MetricsEnum) -> Callable:
        return self._metrics_implementations[metric_id]

    def metric_class_by_id(self, metric_id: MetricsEnum) -> Metric:
        return self._metrics_implementations[metric_id].__self__()
