from typing import Optional
from pymonad.either import Either
import torch

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.industrial.core.operation.transformation.basis.abstract_basis import \
    BasisDecompositionImplementation
from fedot.industrial.core.repository.constanst_repository import \
    SPECTRUM_ESTIMATORS_TORCH, DEFAULT_ESTIMATOR_PARAMETERS
from fedot.industrial.core.architecture.preprocessing.data_convertor import \
    DataConverter, NumpyConverter


class FourierBasisImplementationTorch(BasisDecompositionImplementation):
    """
    A PyTorch-based implementation for Fourier basis decomposition of time
    series data.

    This class provides methods for decomposing time series data using Fourier
    analysis, including power spectral density (psd) estimation and feature
    extraction.

    Attributes:
        threshold (float): Threshold for filtering dominant frequencies.
        Defaults to 0.9.
        output_format (str): Format of the output ('signal' or 'spectrum').
        Defaults to 'signal'.
        approximation (str): Approximation method ('smooth' or 'exact').
        Defaults to 'smooth'.
        min_rank (int): Minimum rank for spectral estimation. Defaults to 5.
        estimator_name (str): Name of the spectral estimator: 'eigen' or
        'non_parametric' (speriodogram). Defaults to 'eigen'.
        estimator: Function for spectral estimation.
        estimator_params (dict): Parameters for the spectral estimator.
        return_feature_vector (bool): Flag to return heuristic feature vector.
        Defaults to False.

    Example:
        x_train = np.random.rand(100, 3, 100)
        y_train = np.random.rand(100).reshape(-1, 1)
        input_data = init_input_data(x_train, y_train)
        input_data.features = torch.tensor(input_data.features,
                                           dtype=torch.float64)
        basis = FourierBasisImplementationTorch({})._transform(input_data)
    """

    def __repr__(self):
        return 'FourierBasisImplementation'

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.threshold = params.get('threshold', 0.9)
        self.output_format = params.get('output_format', 'signal')
        self.approximation = params.get('approximation', 'smooth')
        self.min_rank = params.get('low_rank', 5)
        self.estimator_name = params.get('estimator', 'eigen')
        self.estimator = SPECTRUM_ESTIMATORS_TORCH[self.estimator_name]
        self.estimator_params = params.get('estimator_parameters',
                                           DEFAULT_ESTIMATOR_PARAMETERS[self.estimator_name])
        self.return_feature_vector = params.get(
            'compute_heuristic_representation', False)

        self.logging_params.update({'threshold': self.threshold})

    def _compute_heuristic_features(self, input_data: torch.Tensor):
        """
        Compute heuristic features from the power spectral density of the input
        data.

        Args:
            input_data (torch.Tensor): Input tensor for which to compute
            heuristic features.

        Returns:
            torch.Tensor: Tensor containing heuristic features (mean, variance,
            RMS, peak value, peak frequency, energy, crest factor).
        """
        periodogram_fn = SPECTRUM_ESTIMATORS_TORCH['non_parametric']
        psd = self._get_psd(periodogram_fn,
                            input_data,
                            DEFAULT_ESTIMATOR_PARAMETERS['non_parametric'])

        fft_mean = psd.mean()
        fft_var = psd.var()
        fft_rms = torch.sqrt(torch.mean(psd ** 2))
        fft_peak_value = psd.max()
        fft_peak_freq = psd[torch.argmax(psd)]
        fft_energy = psd.sum()
        fft_crest_factor = fft_peak_value / fft_rms

        feature_vector = torch.stack([
            fft_mean,
            fft_var,
            fft_rms,
            fft_peak_value,
            fft_peak_freq,
            fft_energy,
            fft_crest_factor,
        ])

        return torch.round(feature_vector, decimals=3)

    def _get_psd(self,
                 estimator,
                 input_data: torch.Tensor,
                 estimator_params: None | dict):
        """
        Get the power spectral density (PSD) of the input data using the
        specified estimator.

        Args:
            estimator: Function to estimate the PSD.
            input_data (torch.Tensor): Input tensor for PSD estimation.
            estimator_params (Optional[dict]): Parameters for the estimator.

        Returns:
            torch.Tensor: Power spectral density of the input data.
        """
        if estimator_params is None:
            estimator_params = {}
        try:
            psd = estimator(input_data, **estimator_params)
        except AssertionError:
            old_min_rank = self.min_rank
            self.min_rank = max(1, self.min_rank // 2)
            self.log.info(
                f'Value of min_rank changed from {old_min_rank} to {self.min_rank}'
            )
            if "IP" in list(estimator_params.keys()):
                estimator_params["IP"] = self.min_rank
            psd = estimator(input_data, **estimator_params)
        return psd

    def _tensor_decompose(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decompose the input tensors into their Fourier basis representations.

        Args:
            features (torch.Tensor): Input series with shape of
            (batch_series, chanels, series_length) to decompose.

        Returns:
            torch.Tensor: Decomposed basises of tensors or their heuristic
            features.
        """
        number_of_dim = list(range(features.shape[1]))
        one_dim_predict = features.shape[1] == 1
        if one_dim_predict:
            basis = [
                self._transform_one_sample(series=sample.squeeze(dim=0))
                for sample in features[:, 0, :]
            ]
            basis = torch.stack(basis, dim=0).squeeze(dim=1)

        else:
            basis = [
                [
                    self._transform_one_sample(series=sample.squeeze(dim=0))
                    for sample in features[:, dim, :]
                ]
                for dim in number_of_dim
            ]
            basis = torch.stack(
                [torch.stack(dim_feats) for dim_feats in basis]
            ).transpose(0, 1).squeeze(dim=2)
        return basis

    def _convert_basis_to_predict(self,
                                  basis: torch.Tensor,
                                  input_data: InputData):
        """
        Convert the basis tensor into a prediction format suitable for the task.

        Args:
            basis (torch.Tensor): Basis tensor to be converted into prediction
            format.
            input_data (InputData): Input data containing features, task
            parameters, and other metadata.

        Returns:
            OutputData: An object containing the prediction, input features,
            task parameters, and other metadata.
        """
        # TODO: make NumpyConverter method as general for whole basis methods
        if input_data.features.shape[0] == 1 and input_data.features.dim() == 3:
            self.predict = basis.unsqueeze(0)
        else:
            self.predict = basis

        if input_data.task.task_params is None:
            input_data.task.task_params = self.__repr__()
        elif input_data.task.task_params not in [self.__repr__(),
                                                 'LargeFeatureSpace']:
            input_data.task.task_params.feature_filter = self.__repr__()

        predict = OutputData(idx=input_data.idx,
                             features=input_data.features,
                             predict=self.predict,
                             task=input_data.task,
                             target=input_data.target,
                             data_type=DataTypesEnum.table,
                             supplementary_data=input_data.supplementary_data)
        return predict

    def _transform(self,
                   input_data: InputData,) -> OutputData:
        """
        Transform input data into a Fourier basis representation.

        This method converts the input data into a suitable format, applies
        tensor decomposition, and converts the resulting basis into a prediction
        format.

        Args:
            input_data (InputData): Input data containing features, task
        parameters, and other metadata.

        Returns:
            OutputData: An object containing the prediction, input features,
            task parameters, and other metadata.
        """
        features = DataConverter(data=input_data).convert_to_monad_data()
        features = NumpyConverter(data=features,
                                  to_numpy_array=False).convert_to_torch_format()
        basis = Either.insert(features).then(self._tensor_decompose).value
        predict = self._convert_basis_to_predict(basis=basis,
                                                 input_data=input_data)
        return predict

    def _transform_one_sample(self, series: torch.Tensor) -> torch.Tensor:
        """
        Transform a single time series into its filtered Fourier representation.

        This method processes a single time series by first ensuring it is
        one-dimensional. It then computes the power spectral density (PSD) and
        applies a threshold to filter out non-dominant frequencies. Depending on
        the configuration, it can return either the filtered spectrum or the
        inverse Fourier transform of the filtered spectrum.

        Steps:
        1. Ensure the input series is one-dimensional by squeezing if necessary.
        2. If configured, return heuristic features computed from the PSD.
        3. Compute the PSD of the series.
        4. Apply a threshold to identify dominant frequencies.
        5. Filter the PSD based on the approximation method ('exact' or
        'smooth').
        6. Return either the filtered spectrum or the inverse Fourier transform
        of the filtered spectrum.

        Args:
            series (torch.Tensor): Input time series tensor of shape
            (sequence_length,) or (1, sequence_length).

        Returns:
            torch.Tensor: Filtered spectrum or time-domain signal.
                - If `output_format` is 'spectrum',
                returns the filtered PSD of shape (sequence_length/2 + 1,).
                - If `output_format` is 'signal', returns the inverse Fourier
                transform of the filtered PSD of shape (1, sequence_length).
                - If `return_feature_vector` is True, returns heuristic features
                of shape (7,).
        """
        if series.dim() > 1:
            series = series.squeeze(dim=0)
        if self.return_feature_vector:
            return self._compute_heuristic_features(series)
        psd = self._get_psd(self.estimator, series, self.estimator_params)
        threshold_value = torch.quantile(psd, self.threshold)
        dominant_freq = psd >= threshold_value

        if self.approximation == 'exact':
            psd = psd * (~dominant_freq)
        else:
            psd = psd * dominant_freq

        if self.output_format == 'spectrum':
            filtered_signal = psd
        else:
            filtered_signal = torch.fft.irfft(psd).unsqueeze(0)
        return filtered_signal
