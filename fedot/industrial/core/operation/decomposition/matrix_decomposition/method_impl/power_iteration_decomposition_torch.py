import math
import torch
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.industrial.core.repository.constanst_repository import \
    DEFAULT_SVD_SOLVER_TORCH, DEFAULT_QR_SOLVER_TORCH


def johnson_lindenstrauss_min_dim(n_samples: float, *,
                                  eps: torch.Tensor) -> torch.Tensor:
    """Compute the minimum number of dimensions required for
    Johnson-Lindenstrauss lemma.

    Args:
        n_samples (torch.Tensor): Number of samples.
        eps (float): Tolerance parameter.

    Returns:
        torch.Tensor: Minimum number of dimensions.
    """
    denominator = (eps**2 / 2) - (eps**3 / 3)
    return torch.ceil(4 * math.log(n_samples) / denominator).to(torch.int64)


class RSVDDecompositionTorch:
    """Randomized SVD decomposition with power iteration method.

    Implements the block Krylov subspace method for computing the SVD of a
    matrix with a low computational cost. The method is based on the power
    iteration procedure, which allows us to obtain a low-rank approximation of
    the matrix. The method is based on the following steps:

    1. Random projection of the matrix.
    2. Transformation of the initial matrix to the Gram matrix.
    3. Power iteration procedure.
    4. Orthogonalization of the resulting "sampled" matrix.
    5. Projection of the initial Gram matrix on the new basis obtained from the
    "sampled matrix".
    6. Classical svd decomposition with the chosen type of spectrum
    thresholding.
    7. Compute matrix approximation and choose a new low_rank.
    8. Return matrix approximation.

    Attributes:
        rank (int): Rank of the matrix approximation.
        poly_deg (int): Polynomial degree for power iteration procedure.
        sampling_share (float): Percent of sampling columns. Defaults to 0.7.
        tolerance (list): Tolerance values for Johnson-Lindenstrauss lemma.
        lb_for_sampling_regime (int): Lower bound for considering a matrix as
        large.
        lb_ratio_for_tall_matrix (int): Lower bound ratio for considering a
        matrix as tall.
        is_matrix_big (bool): Flag indicating if the matrix is large.
        is_matrix_tall (bool): Flag indicating if the matrix is tall.
        big_tall_matrix (bool): Flag indicating if the matrix is both big and
        tall.
        projection_rank (int): Rank for random projection.
        random_projection (torch.Tensor): Random projection matrix.
        sampled_tensor_orto (torch.Tensor): Orthogonalized sampled tensor.

    Args:
        params (Optional[OperationParameters]): Dictionary with parameters for
        the operation.
            Expected keys: 'rank', 'power_iter', 'sampling_share', 'tolerance'.
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.rank = params.get('rank', 1)
        self.poly_deg = params.get('power_iter', 3)
        self.sampling_share = params.get('sampling_share', 0.7)
        self.tolerance = params.get('tolerance', [0.5, 0.1])

    def _init_sampling_params(self, tensor: torch.Tensor):
        self.lb_for_sampling_regime = 10000
        self.lb_ratio_for_tall_matrix = 50
        self.projection_rank = math.ceil(min(tensor.shape) *
                                         self.sampling_share)

        self.is_matrix_big = any(dim_len > self.lb_for_sampling_regime
                                 for dim_len in tensor.shape)
        min_dim = min(tensor.shape)
        max_dim = max(tensor.shape)
        self.is_matrix_tall = (max_dim / min_dim > self.lb_ratio_for_tall_matrix
                               if min_dim != 0 else False)
        self.big_tall_matrix = self.is_matrix_big and self.is_matrix_tall

    def _get_stable_rank(self, matrix: torch.Tensor) -> int:
        """
        Compute the stable rank for the CUR decomposition.
        It must be at least 4 times the rank of the matrix but not
        greater than the number of rows or columns of the matrix.

        Args:
            matrix (torch.Tensor): The matrix to decompose.

        Returns:
            int: The stable rank.
        """
        n_samples = max(matrix.shape)
        min_num_samples = johnson_lindenstrauss_min_dim(torch.tensor(n_samples),
                                                        eps=self.tolerance).tolist()
        return max([x if x < n_samples else n_samples for x in min_num_samples])

    def _init_random_params(self, tensor: torch.Tensor):
        """
        Initialize random parameters for projection based on the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor for which to initialize random
            parameters.
        """
        self._init_sampling_params(tensor)
        # Create random matrix for projection
        if self.is_matrix_big:
            self.projection_rank = self._get_stable_rank(tensor)
        self.random_projection = torch.randn(tensor.shape[1],
                                             self.projection_rank,
                                             device=tensor.device,
                                             dtype=tensor.dtype)
        self.big_tall_matrix = all([self.is_matrix_big, self.is_matrix_tall])

    def compute_approximation(self, original_tensor: torch.Tensor,
                              approx_params: dict) -> tuple:
        """Compute the approximation of the original tensor using the given
        approximation parameters.

        Args:
            original_tensor (torch.Tensor): Original tensor to approximate.
            approx_params (dict): Dictionary with approximation parameters.
                Expected keys: 'left_eigenvectors' and 'rank'.

        Returns:
            tuple: Ut, St, Vt from the SVD of the reconstructed matrix.
        """
        Ut_ = approx_params['left_eigenvectors'][:, :approx_params['rank']]
        tensor_approx = self.sampled_tensor_orto @ Ut_
        reconstr_m = tensor_approx @ tensor_approx.T @ original_tensor
        Ut, St, Vt = DEFAULT_SVD_SOLVER_TORCH(reconstr_m, full_matrices=False)
        return Ut, St, Vt

    def decompose(self, tensor: torch.Tensor) -> tuple:
        """
        Block Krylov subspace method for computing the SVD of a matrix with low
        computational cost.

        This method performs the following steps:
        1. Initialize random matrix parameters.
        2. Transform the initial matrix to the Gram matrix.
        3. Perform power iteration procedure.
        4. Orthogonalize the resulting "sampled" matrix.
        5. Project the initial Gram matrix on the new basis obtained from the
        "sampled matrix".
        6. Perform classical SVD decomposition.

        Args:
            tensor (torch.Tensor): Matrix to decompose.

        Returns:
            tuple: Ut, St, Vt decomposition.
        """
        # Initialize random matrix params.
        self._init_random_params(tensor)

        # Transform initial matrix to Gram matrix.
        if self.big_tall_matrix:
            tensor_row_sampled = self.random_projection @ tensor
            grammian = tensor_row_sampled @ tensor_row_sampled.T
            grammian_with_good_spectrum = torch.linalg.matrix_power(grammian,
                                                                    self.poly_deg)
            sampled_tensor = grammian_with_good_spectrum @ tensor_row_sampled
        else:
            grammian = tensor @ tensor.T
            sampled_tensor = torch.linalg.matrix_power(grammian,
                                                       self.poly_deg) @ tensor @ self.random_projection

        # Orthogonalization of the resulting "sampled" matrix.
        self.sampled_tensor_orto, _ = DEFAULT_QR_SOLVER_TORCH(sampled_tensor,
                                                              mode='reduced')

        # Project initial Gramm matrix on new basis.
        M = self.sampled_tensor_orto.T @ grammian @ self.sampled_tensor_orto

        # Sixth step. Classical SVD decomposition.
        Ut, St, Vt = DEFAULT_SVD_SOLVER_TORCH(M, full_matrices=False)

        return Ut, St, Vt
