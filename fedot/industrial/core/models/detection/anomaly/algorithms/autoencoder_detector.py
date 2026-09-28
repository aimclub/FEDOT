from abc import abstractmethod
from typing import Optional

from fedot.core.data.input_data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.preprocessing import StandardScaler
from torch import cuda, device

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.models.detection.anomaly_detector import AnomalyDetector

device = device("cuda:0" if cuda.is_available() else "cpu")


class AutoEncoderDetector(AnomalyDetector):
    """A reconstruction autoencoder model to detect anomalies
    in timeseries data using reconstruction error as an anomaly score.


    Args:
        params: additional parameters for an encapsulated autoencoder model

            .. details:: Possible parameters:

                    - ``learning_rate`` -> learning rate for an optimizer
                    - ``ucl_quantile`` -> upper control limit quantile
                    - ``n_steps_share`` -> share of an n_steps to define n_steps window
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.learning_rate = self.params.get('learning_rate', 0.001)
        self.ucl_quantile = self.params.get('ucl_quantile', 0.99)
        self.n_steps_share = self.params.get('n_steps_share', 0.15)
        self.transformation_mode = 'full'
        self.anomaly_threshold = None
        self.scaler = StandardScaler()

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    def _detect_anomaly_sample(self, score_matrix_row):
        outlier_score = score_matrix_row[0]
        anomaly_sample = outlier_score > self.upper_bound
        return anomaly_sample

    def fit(self, input_data: InputData) -> None:
        self.n_steps = round(input_data.features.shape[0] * self.n_steps_share)
        converted_input_data = self.convert_input_data(input_data)
        self.model_impl = self.build_model()
        self.model_impl.fit(converted_input_data)
        train_prediction = self.model_impl.predict(converted_input_data)
        residual = np.abs(converted_input_data - train_prediction)
        loss = np.mean(residual, axis=1).sum(axis=1)
        self.upper_bound = np.quantile(loss, self.ucl_quantile)  # * 4 / 3

    def convert_input_data(self, input_data: InputData, fit_stage: bool = True) -> np.ndarray:
        if fit_stage:
            values = self.scaler.fit_transform(input_data.features)
        else:
            values = self.scaler.transform(input_data.features)
        output = []
        for i in range(len(values) - self.n_steps + 1):
            output.append(values[i: (i + self.n_steps)])
        return np.stack(output)
