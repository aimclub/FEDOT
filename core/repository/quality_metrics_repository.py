from enum import Enum
from typing import Callable

from core.composer.metrics import F1Metric, MaeMetric, RmseMetric, RocAucMetric, StructuralComplexityMetric


class MetricsEnum(Enum):
    pass


class QualityMetricsEnum(MetricsEnum):
    pass


class ComplexityMetricsEnum(MetricsEnum):
    node_num = 'node_number'
    structural = 'structural'


class ClassificationMetricsEnum(QualityMetricsEnum):
    ROCAUC = 'roc_auc'
    precision = 'precision'
    f1 = 'f1'


class RegressionMetricsEnum(QualityMetricsEnum):
    RMSE = 'rmse'
    MAE = 'mae'


class MetricsRepository:
    __metrics_implementations = {
        ClassificationMetricsEnum.ROCAUC: RocAucMetric.get_value,
        RegressionMetricsEnum.MAE: MaeMetric.get_value,
        RegressionMetricsEnum.RMSE: RmseMetric.get_value,
        ClassificationMetricsEnum.f1: F1Metric.get_value,
        ComplexityMetricsEnum.structural: StructuralComplexityMetric.get_value
    }

    def metric_by_id(self, metric_id: MetricsEnum) -> Callable:
        return self.__metrics_implementations[metric_id]
