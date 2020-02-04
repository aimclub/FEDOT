from enum import Enum


class MetricsEnum(Enum):
    pass


class QualityMetrics(MetricsEnum):
    pass


class ComplexityMetrics(MetricsEnum):
    node_num = "node_number"


class ClassificationMetrics(QualityMetrics):
    ROCAUC = "roc_auc"
    precision = "precision"


class RegressionMetrics(QualityMetrics):
    RMSE = "rmse"
    MAE = "mae"


class MetricsRepository:
    # TODO fill with real implementations
    metrics_implementations = {
        ClassificationMetrics.ROCAUC: object,
        RegressionMetrics.MAE: object,
        RegressionMetrics.RMSE: object,
    }

    def obtain_metric_implementation(self, metric_id):
        return self.metrics_implementations[metric_id]()
