from typing import Optional

from fedot.core.data.input_data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.preprocessing import MinMaxScaler

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix


class SingularSpectrumTransformation:
    """SingularSpectrumTransformation class.

    Change point detection with Singular Spectrum Transformation.

    Note:
        In case of 1D time series to find appropriate hyperparameters value
         for ts_window_length and trajectory_window_length
         are recommended to use WSSAlgorithms.py, WindowSizeSelection class.

    Parameters:
        model_hyperparams:
                        n_components: Number of principal components to keep from
                            functional principal component analysis. Defaults to 3.
                        window_length: Regularization object to be applied.
                        trajectory_window_length: .
    Attributes:

    """

    def __init__(self, params: Optional[OperationParameters] = None):

        self.n_components = params.get('n_components', 2)
        self.length_of_detection_window = params.get('window_length', 30)
        self.trajectory_window_length = params.get(
            'trajectory_window_length', 10)
        self.dynamic_mode = params.get('n_components', False)
        self.intersection_lag = params.get('delay_lag', 5)

    def _scale_ts(self, time_series: np.ndarray):
        time_series_scaled = MinMaxScaler(feature_range=(1, 2)) \
            .fit_transform(time_series.reshape(-1, 1))[:, 0]
        return time_series_scaled

    def _online_detection(self,
                          train_features,
                          start_idx,
                          end_idx):
        self.model = []
        test_features = []
        for t in range(start_idx, end_idx):
            start_idx_hist, end_idx_hist = t - self.window_length - self.lag, t - self.lag
            start_idx_test, end_idx_test = t - self.window_length, t
            x_history = HankelMatrix(time_series=train_features[start_idx_hist:end_idx_hist],
                                     window_size=self.trajectory_window_length)
            x_test = HankelMatrix(time_series=train_features[start_idx_test:end_idx_test],
                                  window_size=self.trajectory_window_length)
            self.model.append(x_history)
            test_features.append(x_test)
        return test_features

    def fit(self, input_data: InputData) -> list:
        if len(input_data.features.shape) >= 3:
            list_of_history, list_of_current = [], []
            for time_series in input_data.features:
                history_state, current_state = self._fit(
                    train_features=time_series)
                list_of_history.append(history_state)
                list_of_current.append(current_state)
            return list_of_history, list_of_current
        else:
            return self._fit(train_features=input_data.features)

    def _fit(self, train_features: np.ndarray) -> list:
        """Core implementation of offline score calculation.

        Args:
            self.dynamic_mode: mode for SST metrics calculation.
            train_features: input 1D time series data.

        Returns
            score: 1d array change point score with 1 and 0 if view True.

        """

        self.length_of_detection_window = round(
            train_features.shape[0] * (self.length_of_detection_window / 100))
        self.trajectory_window_length = round(
            self.length_of_detection_window * (self.trajectory_window_length / 100))
        self.intersection_lag = round(
            self.trajectory_window_length * (self.intersection_lag / 100))
        end_of_train_data, end_of_test_data = self.length_of_detection_window + \
            self.intersection_lag + 1, train_features.shape[0] + 1

        if not self.dynamic_mode:
            test_features = []
            self.model = HankelMatrix(time_series=train_features[:end_of_train_data],
                                      window_size=self.trajectory_window_length)
            for t in range(end_of_train_data, end_of_test_data):
                start_idx_test, end_idx_test = t - self.intersection_lag, t
                x_test = HankelMatrix(time_series=train_features[start_idx_test:end_idx_test],
                                      window_size=self.trajectory_window_length)
                test_features.append(x_test)
        else:
            test_features = self._online_detection(
                train_features, end_of_train_data, end_of_test_data)

        return test_features

    def predict(self, input_data: InputData) -> list:
        if len(input_data.features.shape) > 3:
            residual_list = []
            for history_data, target_data in zip(
                    input_data.features, input_data.target):
                residual = self._predict(history_data, target_data)
                residual_list.append(residual)
            return residual_list
        else:
            return self._predict(test_features=input_data.features)

    def _predict(
            self,
            test_features: HankelMatrix,
            target: HankelMatrix) -> list:
        """Core implementation of offline score calculation.

        Args:
            self.dynamic_mode: mode for SST metrics calculation.
            _x: input 1D time series data.

        Returns
            score: 1d array change point score with 1 and 0 if view True.
            :param target:
            :param test_features:

        """

        if self.dynamic_mode:
            def dynamic_sst(trajectory_data_tuple):
                return self._sst_svd(
                    trajectory_data_tuple[0].trajectory_matrix,
                    trajectory_data_tuple[1].trajectory_matrix)

            trajectory_data_tuple = list(zip(test_features, target))
            score_list = list(map(dynamic_sst, trajectory_data_tuple))
        else:
            def static_sst(x_history):
                return self._sst_svd(
                    self.model.trajectory_matrix, x_history.trajectory_matrix)

            score_list = list(map(static_sst, test_features))
        residual = np.diff(score_list)
        return residual

    def _sst_svd(self, x_test: list = None, x_history: list = None) -> float:
        """Singular value decomposition to count distance score between matrix

        Args:
            x_test: current matrix of features
            x_history: historical matrix of features

        Returns
            1 - s[0]: distance score between two matrix

        """
        u_test, s_test, _ = np.linalg.svd(x_test, full_matrices=False)
        u_history, s_hist, _ = np.linalg.svd(x_history, full_matrices=False)
        s_cov = u_test[:, :self.n_components].T @ u_history[:,
                                                            :self.n_components]
        u_cov, s, _ = np.linalg.svd(s_cov, full_matrices=False)
        return 1 - s[0]
