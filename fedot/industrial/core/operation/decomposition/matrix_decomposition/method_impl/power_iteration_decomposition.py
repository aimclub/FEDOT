import math
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.random_projection import johnson_lindenstrauss_min_dim

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.repository.constanst_repository import DEFAULT_SVD_SOLVER, DEFAULT_QR_SOLVER


class RSVDDecomposition:
    """Randomized SVD decomposition with power iteration method.
    Implements the block Krylov subspace method for computing the SVD of a matrix with a low computational cost.
    The method is based on the power iteration procedure, which allows us to obtain a low-rank approximation of the
    matrix. The method is based on the following steps:
    1. Random projection of the matrix.
    2. Transformation of the initial matrix to the Gram matrix.
    3. Power iteration procedure.
    4. Orthogonalization of the resulting "sampled" matrix.
    5. Projection of the initial Gram matrix on the new basis obtained from the "sampled matrix".
    6. Classical svd decomposition with the chosen type of spectrum thresholding.
    7. Compute matrix approximation and choose a new low_rank.
    8. Return matrix approximation.

    Args:
        params: dictionary with parameters for the operation:
            rank: rank of the matrix approximation
            power_iter: polynom degree for power iteration procedure
            sampling_share: percent of sampling columns. By default - 70%

    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.rank = params.get('rank', 1)
        # Polynom degree for power iteration procedure.
        self.poly_deg = params.get('power_iter', 3)
        # Percent of sampling columns. By default - 70%
        self.sampling_share = params.get('sampling_share', 0.7)
        self.tolerance = params.get('tolerance', [0.5, 0.1])

    def _init_sampling_params(self, tensor):
        self.lb_for_sampling_regime = 10000
        self.lb_ratio_for_tall_matrix = 50
        self.is_matrix_big = False
        self.projection_rank = math.ceil(min(tensor.shape) * self.sampling_share)
        for dim_len in tensor.shape:
            if dim_len > self.lb_for_sampling_regime:
                self.is_matrix_big = True
                break
        min_dim = min(tensor.shape)
        max_dim = max(tensor.shape)
        self.is_matrix_tall = max_dim / min_dim > self.lb_ratio_for_tall_matrix if min_dim != 0 else False
        self.big_tall_matrix = all([self.is_matrix_big, self.is_matrix_tall])

    def _init_random_params(self, tensor):
        self._init_sampling_params(tensor)
        # Create random matrix for projection
        if self.is_matrix_big:
            self.projection_rank = self._get_stable_rank(tensor)
        self.random_projection = np.random.randn(tensor.shape[1], self.projection_rank)
        self.big_tall_matrix = all([self.is_matrix_big, self.is_matrix_tall])

    def _get_stable_rank(self, matrix):
        """
        Compute the stable rank for the CUR decomposition.
        It must be at least 4 times the rank of the matrix but not
        greater than the number of rows or columns of the matrix.

        Args:
            matrix: the matrix to decompose.

        Returns:
            the stable rank
        """
        n_samples = max(matrix.shape)
        min_num_samples = johnson_lindenstrauss_min_dim(n_samples, eps=self.tolerance).tolist()
        return max([x if x < n_samples else n_samples for x in min_num_samples])

    def compute_approximation(self, original_tensor, approx_params: dict):
        Ut_ = approx_params['left_eigenvectors'][:, :approx_params['rank']]
        tensor_approx = self.sampled_tensor_orto @ Ut_
        reconstr_m = tensor_approx @ tensor_approx.T @ original_tensor
        Ut, St, Vt = DEFAULT_SVD_SOLVER(reconstr_m, full_matrices=False)
        return Ut, St, Vt

    def decompose(self,
                  tensor: np.array) -> list:
        """Block Krylov subspace method for computing the SVD of a matrix with a low computational cost.

        Args:
            tensor: matrix to decompose
            approximation: if True, the matrix approximation will be computed
            regularized_rank: rank of the matrix approximation
            reg_type: type of regularization. 'hard_thresholding' or 'explained_dispersion'

        Returns:
            u, s, vt: decomposition

        """
        # First step. Initialize random matrix params.
        self._init_random_params(tensor)
        # Second step. Transform initial matrix to Gram. matrix

        if self.big_tall_matrix:
            tensor_row_sampled = self.random_projection @ tensor  # For tall and big matrix we use "row-sampling" operator
            grammian = tensor_row_sampled @ tensor_row_sampled.T
            grammian_with_good_spectrum = np.linalg.matrix_power(grammian, self.poly_deg)
            sampled_tensor = grammian_with_good_spectrum @ tensor_row_sampled
        else:
            # Third step. Power iteration procedure. First we raise the Gram matrix to the chosen degree.
            # This step is necessary in order to obtain a more "pronounced" spectrum (in which the eigenvalues
            # are well separated from each other). The important point is that the exponentiation procedure only changes
            # the eigenvalues but does not change the eigenvectors. Next, the resulting matrix is multiplied with the
            # original matrix ("overweightning" the column space) and then multiplied with a random matrix
            # in order to reduce the dimension and facilitate the procedure for
            # "large" matrices.
            grammian = tensor @ tensor.T
            sampled_tensor = np.linalg.matrix_power(grammian, self.poly_deg) @ tensor @ self.random_projection
        # Fourth step. Orthogonalization of the resulting "sampled" matrix
        # creates for us a basis of eigenvectors.
        self.sampled_tensor_orto, _ = DEFAULT_QR_SOLVER(sampled_tensor, mode='reduced')
        # Fifth step. Project initial Gramm matrix on new basis obtained
        # from "sampled matrix".
        M = self.sampled_tensor_orto.T @ grammian @ self.sampled_tensor_orto
        # Six step. Classical svd decomposition with choosen type of
        # spectrum thresholding
        Ut, St, Vt = DEFAULT_SVD_SOLVER(M, full_matrices=False)

        return Ut, St, Vt
