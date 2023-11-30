from typing import Dict, Optional, Protocol, TypeVar, Union

from golem.utilities.data_structures import ComparableEnum as Enum

from fedot.core.composer.metrics import (Accuracy, ComputationTime, F1, Logloss, MAE, MAPE, MASE, MSE, MSLE, Metric,
                                         NodeNum, Precision, R2, RMSE, ROCAUC, SMAPE, Silhouette, StructuralComplexity)
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline


class MetricsEnum(Enum):
    def __str__(self):
        return self.value

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


NumberType = Union[int, float, complex]
PipelineType = TypeVar('PipelineType', bound=Pipeline, covariant=True)


class QualityMetricCallable(Protocol):
    def __call__(self, pipeline: PipelineType, reference_data: InputData,
                 validation_blocks: Optional[int] = None) -> NumberType: ...


class ComplexityMetricCallable(Protocol):
    def __call__(self, pipeline: PipelineType) -> NumberType: ...


MetricCallable = Union[QualityMetricCallable, ComplexityMetricCallable]
MetricType = Union[MetricCallable, MetricsEnum]


class QualityMetricsEnum(MetricsEnum):
    pass


class ComplexityMetricsEnum(MetricsEnum):
    node_number = 'node_number'
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


class TimeSeriesForecastingMetricsEnum(QualityMetricsEnum):
    MASE = 'mase'
    RMSE = 'rmse'
    MSE = 'mse'
    MSLE = 'neg_mean_squared_log_error'
    MAPE = 'mape'
    SMAPE = 'smape'
    MAE = 'mae'
    R2 = 'r2'
    RMSE_penalty = 'rmse_pen'


class MetricsRepository:
    _metrics_implementations: Dict[MetricsEnum, MetricCallable] = {
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

        # ts forecasting
        TimeSeriesForecastingMetricsEnum.MASE: MASE.get_value,

        # clustering
        ClusteringMetricsEnum.silhouette: Silhouette.get_value,

        # structural
        ComplexityMetricsEnum.structural: StructuralComplexity.get_value,
        ComplexityMetricsEnum.node_number: NodeNum.get_value,
        ComplexityMetricsEnum.computation_time: ComputationTime.get_value
    }

    _metrics_classes = {metric_id: getattr(metric_func, '__self__')
                        for metric_id, metric_func in _metrics_implementations.items()}

    @staticmethod
    def get_metric(metric_id: MetricsEnum) -> MetricCallable:
        return MetricsRepository._metrics_implementations[metric_id]

    @staticmethod
    def get_metric_class(metric_id: MetricsEnum) -> Metric:
        return MetricsRepository._metrics_classes[metric_id]
