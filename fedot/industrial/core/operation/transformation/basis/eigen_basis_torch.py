from pymonad.either import Either
from typing import Optional
import torch

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.industrial.core.architecture.preprocessing.data_convertor import DataConverter, NumpyConverter
from fedot.industrial.core.operation.decomposition.matrix_decomposition.decomposer import MatrixDecomposerTorch
from fedot.industrial.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation
from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix
from fedot.industrial.core.operation.transformation.regularization.spectrum import reconstruct_basis_torch


class EigenBasisImplementationTorch(BasisDecompositionImplementation):
    """
    A PyTorch-based implementation for eigenbasis decomposition of time series
    data.

    This class provides methods for decomposing time series data using singular
    value decomposition (SVD) and other related techniques. It supports rank
    regularization and is optimized for GPU acceleration.

    Attributes:
        window_size (int): The size of the window used for decomposition.
        Defaults to 20.
        decomposition_type (str): The type of decomposition to use. 'svd',
        'random_svd', 'cur' are available. Defaults to 'svd'.
        rank_regularization (str): The method for rank regularization.
        'explained_dispersion', 'hard_thresholding', 'knee_point' are available.
        Defaults to 'hard_thresholding'.
        logging_params (dict): Dictionary for logging parameters, includes
        window size.
        explained_dispersion (float or None): The explained dispersion of the
        decomposition. Defaults to None.
        SV_threshold (int or None): The threshold for singular values.
        Defaults to None.
        solver (MatrixDecomposerTorch): The matrix decomposer solver instance
        for performing the decomposition.

    Example:
        x_train = np.random.rand(100, 3, 100)
        y_train = np.random.rand(100).reshape(-1, 1)
        input_data = init_input_data(x_train, y_train)
        input_data.features = torch.tensor(input_data.features,
                                           dtype=torch.float64)
        basis = EigenBasisImplementationTorch({})._transform(input_data)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = self.params.get('window_size', 20)
        self.decomposition_type = params.get('decomposition_type', 'svd')
        self.rank_regularization = self.params.get('rank_regularization',
                                                   'hard_thresholding')
        self.logging_params.update({'WS': self.window_size})
        self.explained_dispersion = None
        self.SV_threshold = None
        self.solver = MatrixDecomposerTorch(
            {'decomposition_type': self.decomposition_type,
             'decomposition_params': {'spectrum_regularization':
                                      self.rank_regularization}})

    def __repr__(self):
        return 'EigenBasisImplementation'

    def get_threshold(self, data: torch.Tensor) -> int:
        """
        Compute the most common SVD rank across signals using Dask.

        Args:
            data (torch.Tensor): Input tensor of shape (N, D, T).

        Returns:
            int: Most frequent rank value.
        """
        number_of_dim = range(data.shape[1])
        one_dim_predict = data.shape[1] == 1

        def mode_func(x):
            return max(set(x), key=x.count)

        # single dimension case
        if one_dim_predict:
            list_of_ranks = [
                self._transform_one_sample(sample, svd_flag=True)
                for sample in data[:, 0, :]
            ]

        #  multi-dimension case
        else:
            to_comp = []
            for dim in number_of_dim:
                dim_ranks = []
                for signal in data[:, dim, :]:
                    r = self._transform_one_sample(signal, svd_flag=True)
                    if isinstance(r, torch.Tensor):
                        r = r.item()
                    dim_ranks.append(r)
                to_comp.append(dim_ranks)
            list_of_ranks = torch.tensor(to_comp).flatten().tolist()

        return mode_func(list_of_ranks)

    def _tensor_decompose(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decompose input tensor into feature representations using SVD-based
        transform.

        Args:
            features (torch.Tensor): Input tensor of shape (N, D, T).

        Returns:
            torch.Tensor: SVD-based transformation.
        """

        number_of_dim = list(range(features.shape[1]))
        one_dim_predict = features.shape[1] == 1

        # threshold computation
        if self.SV_threshold is None:
            self.SV_threshold = self.get_threshold(data=features)
            self.logging_params.update({'SV_thr': self.SV_threshold})

        # single dimension case
        if one_dim_predict:
            feature_matrix = [
                self._transform_one_sample(sample)
                for sample in features[:, 0, :]
            ]
            feature_matrix = torch.stack(feature_matrix, dim=0).squeeze(dim=1)

        # multi-dimension case
        else:
            feature_matrix = [
                [
                    self._transform_one_sample(sample)
                    for sample in features[:, dim, :]
                ]
                for dim in number_of_dim
            ]
            feature_matrix = torch.stack(
                [torch.stack(dim_feats) for dim_feats in feature_matrix]
            ).transpose(0, 1)

        return feature_matrix

    def _convert_basis_to_predict(self,
                                  basis: torch.Tensor,
                                  input_data: InputData):
        """
        Convert the basis tensor into a prediction format suitable for the task.

        This function adjusts the shape of the basis tensor to match the
        expected prediction format, and updates the task parameters in the input
        data if necessary.

        Args:
            basis (torch.Tensor): Basis tensor to be converted into prediction
            format.
            input_data (InputData): Input data containing features, task
            parameters, and other metadata.

        Returns:
            OutputData: An object containing the prediction, input features,
            task parameters, and other metadata. The prediction is stored in the
            `predict` attribute.
        """
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

    def _transform(self, input_data: InputData) -> OutputData:
        """
        Transform input data into a basis representation using decomposition
        techniques.

        This method converts the input data into a suitable format, applies
        tensor decomposition, and converts the resulting basis into a Outputdata
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
        predict = self._convert_basis_to_predict(basis, input_data)
        return predict

    def _get_1d_basis(self, data: dict) -> torch.Tensor:
        """
        Obtain a 1D basis from decomposition results by applying thresholding
        and inverse transformation.

        This method first truncates the spectrum of the decomposition results to
        a predefined threshold, then reconstructs the basis into a 1D format.

        Args:
            data (dict): Dictionary containing decomposition results with keys
            'left_eigenvectors', 'spectrum', 'right_eigenvectors', and 'rank'.

        Returns:
            torch.Tensor: A 1D basis tensor obtained from the decomposition
            results.
        """
        def inverse_transformation(data):
            return reconstruct_basis_torch(U=data['left_eigenvectors'],
                                           Sigma=data['spectrum'],
                                           VT=data['right_eigenvectors'],
                                           ts_length=self.ts_length)

        def threshold(data):
            data['spectrum'] = data['spectrum'][:self.SV_threshold]
            return data

        basis = (
            Either.insert(data).then(threshold).then(
                inverse_transformation).value
        )
        return torch.transpose(basis, 0, 1)

    def _transform_one_sample(self,
                              series: torch.Tensor,
                              svd_flag: bool = False) -> dict | torch.Tensor:
        """
        Transform a single time series sample into its basis representation.

        This method converts the time series into a trajectory matrix using a
        Hankel matrix transformation, applies decomposition, and optionally
        returns either the rank or the 1D basis.

        Args:
            series (torch.Tensor): Input time series tensor.
            svd_flag (bool): If True, return the rank of the decomposition.
            Otherwise, return the 1D basis. Defaults to False.

        Returns:
            dict | torch.Tensor: If svd_flag is True, returns the rank of the
            decomposition as a dictionary. Otherwise, returns the 1D basis as a
            tensor.
        """
        window_size = round(series.shape[0] * (self.window_size / 100))
        trajectory_transformer = HankelMatrix(
            time_series=series, window_size=window_size)
        data = trajectory_transformer.trajectory_matrix
        self.ts_length = trajectory_transformer.ts_length
        basis = Either.insert(data).then(self.solver.apply).value
        if svd_flag:
            return basis['rank']
        else:
            return self._get_1d_basis(basis)
