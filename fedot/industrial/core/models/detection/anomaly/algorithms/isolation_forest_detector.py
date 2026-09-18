from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.ensemble import IsolationForest as SklearnIsolationForest

from fedot.industrial.core.models.detection.anomaly_detector import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest or iForest builds an ensemble of iTrees for a given data set, then anomalies are those
    instances which have short average path lengths on the iTrees.

    Args:
        params: additional parameters for a IsolationForest model

            .. details:: Possible parameters:

                    - ``random_state`` -> random seed used for reproducibility
                    - ``n_jobs`` -> number of CPU cores to use for parallelism
                    - ``contamination`` -> expected proportion of anomalies in the dataset
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.random_state = self.params.get('random_state', 0)
        self.n_jobs = self.params.get('n_jobs', -1)
        self.contamination = self.params.get('contamination', 'auto')
        self.anomaly_threshold = self.params.get('anomaly_thr', 0.3)
        self.transformation_mode = 'full'

    def build_model(self):
        return SklearnIsolationForest(random_state=self.random_state,
                                      n_jobs=self.n_jobs,
                                      contamination=self.contamination)
