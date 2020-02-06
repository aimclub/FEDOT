from enum import Enum


class MetricsEnum(Enum):
    pass


class QualityMetricsEnum(MetricsEnum):
    pass


class ComplexityMetricsEnum(MetricsEnum):
    node_num = 'node_number'


class ClassificationMetricsEnum(QualityMetricsEnum):
    ROCAUC = 'roc_auc'
    precision = 'precision'


class RegressionMetricsEnum(QualityMetricsEnum):
    RMSE = 'rmse'
    MAE = 'mae'


class MetricsRepository:
    # TODO fill with real implementations
    metrics_implementations = {
        ClassificationMetricsEnum.ROCAUC: object,
        RegressionMetricsEnum.MAE: object,
        RegressionMetricsEnum.RMSE: object,
    }

    def obtain_metric_implementation(self, metric_id: MetricsEnum):
        return self.metrics_implementations[metric_id]()
