from enum import Enum
from typing import Callable

from core.composer.metrics import RmseMetric, StructuralComplexityMetric, MaeMetric, RocAucMetric


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


class RegressionMetricsEnum(QualityMetricsEnum):
    RMSE = 'rmse'
    MAE = 'mae'


class MetricsRepository:
    __metrics_implementations = {
        ClassificationMetricsEnum.ROCAUC: RocAucMetric.get_value,
        RegressionMetricsEnum.MAE: MaeMetric.get_value,
        RegressionMetricsEnum.RMSE: RmseMetric.get_value,
        ComplexityMetricsEnum.structural: StructuralComplexityMetric.get_value
    }

    def metric_by_id(self, metric_id: MetricsEnum) -> Callable:
        return self.__metrics_implementations[metric_id]
