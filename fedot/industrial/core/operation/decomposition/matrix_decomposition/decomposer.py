import torch
from typing import Optional, Union

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.operation.decomposition.matrix_decomposition.method_impl.column_sampling_decomposition_torch import \
    CURDecompositionTorch
from fedot.industrial.core.operation.decomposition.matrix_decomposition.method_impl.power_iteration_decomposition_torch import \
    RSVDDecompositionTorch
from fedot.industrial.core.operation.decomposition.matrix_decomposition.method_impl.column_sampling_decomposition import \
    CURDecomposition
from fedot.industrial.core.operation.decomposition.matrix_decomposition.method_impl.power_iteration_decomposition import \
    RSVDDecomposition
from fedot.industrial.core.operation.decomposition.matrix_decomposition.method_impl.svd_decompostion import \
    SVDDecomposition
from fedot.industrial.core.operation.decomposition.matrix_decomposition.method_impl.svd_decomposition_torch import \
    SVDDecompositionTorch
from fedot.industrial.core.operation.filtration.channel_filtration import _detect_knee_point, _detect_knee_point_torch
from fedot.industrial.core.operation.transformation.regularization.spectrum import singular_value_hard_threshold, \
    sv_to_explained_variance_ratio, sv_to_explained_variance_ratio_torch, singular_value_hard_threshold_torch


class MatrixDecomposerTorch:
    """
    A PyTorch-based matrix decomposer for performing various matrix
    decomposition techniques.

    This class supports multiple decomposition methods such as SVD,
    Randomized SVD, and CUR. It provides functionality for spectrum
    regularization, low-rank approximation, and tensor decomposition.

    Attributes:
        decomposition_type (str): Type of decomposition. Defaults to 'svd'.
        decomposition_params (dict): Parameters for the decomposition method.
        min_components (int or None): Minimum number of components for
        decomposition.
        spectrum_reg (dict): Dictionary of spectrum regularization methods.
        decompose_method (dict): Dictionary of available decomposition methods.
        decomposition_strategy: Instance of the selected decomposition method.
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.decomposition_type = params.get('decomposition_type', 'svd')
        self.decomposition_params = params.get('decomposition_params', 3)
        self.min_components = params.get('min_components_number', None)
        self.spectrum_reg = {
            'explained_dispersion': sv_to_explained_variance_ratio_torch,
            'hard_thresholding': singular_value_hard_threshold_torch,
            'knee_point': _detect_knee_point_torch}
        self.decompose_method = {'svd': SVDDecompositionTorch,
                                 'random_svd': RSVDDecompositionTorch,
                                 'cur': CURDecompositionTorch, }
        self.decomposition_strategy = (
            self.decompose_method[self.decomposition_type]()
        )

    def spectrum_regularization(self,
                                spectrum: torch.Tensor,
                                reg_type: str = 'hard_thresholding') -> list:
        """
        Apply spectrum regularization to the singular values.

        Args:
            spectrum (torch.Tensor): Singular values to be regularized.
            reg_type (str): Type of regularization to apply. Defaults to
            'hard_thresholding'. Available options: 'explained_dispersion',
            'hard_thresholding', 'knee_point'.

        Returns:
            list: Regularized singular values.
        """
        if reg_type in self.spectrum_reg.keys():
            return self.spectrum_reg[reg_type](spectrum)
        else:
            return spectrum.tolist()

    def get_low_rank(self, spectrum: list) -> int:
        """
        Determine the low rank based on the singular values and minimum
        components.

        Args:
            spectrum (list): List of singular values.

        Returns:
            int: The estimated low rank.
        """
        return min(len(spectrum), self.min_components)

    def get_tensor_approximation(self, tensor: Union[dict, torch.Tensor]):
        """
        Compute the tensor approximation using the decomposition strategy.

        Args:
            tensor (Union[dict, torch.Tensor]): Input tensor or dictionary of
            tensors to approximate.

        Returns:
            torch.Tensor: Approximated tensor.
        """
        if isinstance(tensor, dict):
            tensor_approx = (
                self.decomposition_strategy.compute_approximation(*tensor.values())
            )
        else:
            tensor_approx = (
                self.decomposition_strategy.compute_approximation(tensor)
            )
        return tensor_approx

    def apply(self, tensor: torch.Tensor):
        """
        Apply the decomposition to the input matrix.

        This method performs the following steps:
        1. Estimates the lower bound for rank.
        2. Decomposes the original tensor.
        3. Applies spectrum regularization.
        4. Computes the approximation and estimates the stable rank.

        Args:
            tensor (torch.Tensor): Input tensor to decompose.

        Returns:
            dict: Dictionary containing the decomposition results with keys:
            'left_eigenvectors', 'spectrum', 'right_eigenvectors', and 'rank'.
        """
        # Get lower bound for rank estimation. By default - 10 % of all
        # data, at least 2
        if self.min_components is None:
            self.min_components = max(int(min(tensor.shape) / 10), 2)

        # Get a decomposition of original tensor
        U, S, V = self.decomposition_strategy.decompose(tensor)

        # Get spectrum regularization. In case of CUR we dont use it.
        if len(S.shape) != 1:
            stable_rank = self.decomposition_strategy.stable_rank
        else:
            S_reg = (
                self.spectrum_regularization(spectrum=S,
                                             reg_type=self.decomposition_params['spectrum_regularization'])
            )
            stable_rank = self.get_low_rank(S_reg)

        # Get approx data and estimated stable rank.
        result_dict = dict(left_eigenvectors=U,
                           spectrum=S,
                           right_eigenvectors=V,
                           rank=stable_rank)

        # In case of power iteration with random approx we use computed matrix
        # to rotate original tensor in choosen basis.
        if self.decomposition_type.__contains__('random'):
            result_dict['left_eigenvectors'], result_dict['spectrum'], \
                result_dict['right_eigenvectors'] \
                = self.decomposition_strategy.compute_approximation(tensor,
                                                                    result_dict)
        return result_dict


class MatrixDecomposer:
    """
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.decomposition_type = params.get('decomposition_type', 'svd')
        self.decomposition_params = params.get('decomposition_params', 3)
        self.min_components = params.get('min_components_number', None)
        self.spectrum_reg = {'explained_dispersion': sv_to_explained_variance_ratio,
                             'hard_thresholding': singular_value_hard_threshold,
                             'knee_point': _detect_knee_point}
        self.decompose_method = {'svd': SVDDecomposition,
                                 'random_svd': RSVDDecomposition,
                                 'cur': CURDecomposition,
                                 'dmd': None}
        self.decomposition_strategy = self.decompose_method[self.decomposition_type]()

    def spectrum_regularization(self,
                                spectrum: np.array,
                                reg_type: str = 'hard_thresholding'):
        if reg_type in self.spectrum_reg.keys():
            return self.spectrum_reg[reg_type](spectrum)
        else:
            return spectrum

    def get_low_rank(self, spectrum):
        return min(len(spectrum), self.min_components)

    def get_tensor_approximation(self, tensor: Union[dict, np.ndarray]):
        if isinstance(tensor, dict):
            tensor_approx = self.decomposition_strategy.compute_approximation(*tensor.values())
        else:
            tensor_approx = self.decomposition_strategy.compute_approximation(tensor)
        return tensor_approx

    def apply(self, tensor: np.ndarray):
        # Step 1. Get lower bound for rank estimation. By default - 10 % of all data, at least 2
        if self.min_components is None:
            self.min_components = max(int(min(tensor.shape) / 10), 2)
        # Step 2. Get a decomposition of original tensor
        U, S, V = self.decomposition_strategy.decompose(tensor)
        # Step 3. Get spectrum regularization. In case of CUR decomposition we dont use it.
        if len(S.shape) != 1:
            stable_rank = self.decomposition_strategy.stable_rank
        else:
            S_reg = self.spectrum_regularization(spectrum=S,
                                                 reg_type=self.decomposition_params['spectrum_regularization'])
            stable_rank = self.get_low_rank(S_reg)
        # Step 4. Get approx data and estimated stable rank.
        result_dict = dict(left_eigenvectors=U,
                           spectrum=S,
                           right_eigenvectors=V,
                           rank=stable_rank)
        # Step 4.1. In case of power iteration with random approx we use computed matrix
        # to rotate original tensor in choosen basis
        if self.decomposition_type.__contains__('random'):
            result_dict['left_eigenvectors'], result_dict['spectrum'], result_dict['right_eigenvectors'] \
                = self.decomposition_strategy.compute_approximation(tensor, result_dict)
        return result_dict
