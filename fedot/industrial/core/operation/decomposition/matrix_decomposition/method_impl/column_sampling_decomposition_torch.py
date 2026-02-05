from typing import Tuple, Union, Optional
import torch

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.industrial.core.operation.decomposition.matrix_decomposition.method_impl.power_iteration_decomposition_torch import (
    johnson_lindenstrauss_min_dim)

from fedot.industrial.core.repository.constanst_repository import DEFAULT_SVD_SOLVER_TORCH

RANK_REPRESENTATION = Union[int, float]


class CURDecompositionTorch:
    """
    CUR decomposition is a low-rank matrix decomposition method that is based on
    selecting a subset of columns and rows of the original matrix. The method is
    based on the Johnson-Lindenstrauss lemma and is used to approximate the
    original matrix with a low-rank matrix. The CUR decomposition is defined as
    follows: A = C @ U @ R, where A is the original matrix, C is a subset of
    columns of A, U is a subset of rows of A, and R is a subset of rows of A.
    The selection of columns and rows is based on the probabilities p and q,
    which are computed based on the norms of the columns and rows of A. The
    selection of columns and rows is done in such a way that the approximation
    error is minimized.

    Args:
        params: the parameters of the operation
            rank: the rank of the decomposition
            tolerance: the tolerance of the decomposition
            return_samples: whether to return the samples or the decomposition
            matrices
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.stable_rank = params.get('rank', None)
        self.tolerance = params.get('tolerance', torch.Tensor([0.5, 0.1, 0.05]))
        self.return_samples = params.get('return_samples', True)
        self.column_indices = None
        self.row_indices = None
        self.classes_idx = None
        self.column_space = 'Full'
        self.fitted = True

    def _get_stable_rank(self, matrix: torch.Tensor):
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
        min_num_samples = johnson_lindenstrauss_min_dim(n_samples,
                                                        eps=self.tolerance).tolist()
        self.stable_rank = min([x if x < n_samples else n_samples
                                for x in min_num_samples])

    def decompose(self, tensor: torch.Tensor) -> Tuple:
        """
        Perform CUR decomposition on the input tensor.

        CUR decomposition approximates the original matrix using a subset of its
        columns and rows. The method selects columns and rows based on their
        norms and constructs matrices C, U, and R, where C consists of selected
        columns, R consists of selected rows, and U is derived from the
        intersection of selected columns and rows.

        Args:
            tensor (torch.Tensor): Input 2D matrix to decompose.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                A tuple containing three tensors:
                - C: Matrix consisting of selected columns of the original
                tensor.
                - U: Matrix derived from the pseudoinverse of the intersection
                matrix W.
                - R: Matrix consisting of selected rows of the original tensor.
        """
        if self.stable_rank is None:
            self._get_stable_rank(tensor)
        # create sub matrices for CUR-decompostion
        array = torch.clone(tensor)
        c, w, r = self.select_rows_cols(array)
        # evaluate pseudoinverse for W - U^-1
        U, Sigma, VT = DEFAULT_SVD_SOLVER_TORCH(w, full_matrices=False)
        Sigma_plus = torch.linalg.pinv(torch.diag(Sigma))
        # aprox U using pseudoinverse
        u = VT.T @ Sigma_plus @ Sigma_plus @ U.T
        return c, u, r

    def select_rows_cols(self, matrix: torch.Tensor) -> Tuple:
        """Select rows and columns based on their norms.

        Args:
            matrix (torch.Tensor): Input matrix.

        Returns:
            tuple: C_matrix, W_matrix, R_matrix.
        """
        # Replace NaN values with zeros and normalize the matrix
        matrix = torch.nan_to_num(matrix)
        matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-8)

        # Evaluate norms for columns and rows
        col_norms = torch.linalg.norm(matrix, dim=0)
        row_norms = torch.linalg.norm(matrix, dim=1)
        matrix_norm = torch.linalg.norm(matrix, ord='fro')

        # Compute the probabilities for selecting columns and rows
        col_probs = col_norms / matrix_norm
        row_probs = row_norms / matrix_norm

        col_rank, row_rank = self.stable_rank, self.stable_rank

        # Select top columns and rows based on probabilities
        self.column_indices = torch.sort(torch.argsort(col_probs,
                                                       descending=True)[:col_rank]).values
        self.row_indices = torch.sort(torch.argsort(row_probs,
                                                    descending=True)[:row_rank]).values

        if self.classes_idx is not None:
            row_indices_list = []
            for cls_idx in self.classes_idx:
                cls_row_probs = row_probs[cls_idx]
                cls_row_indices = torch.argsort(cls_row_probs,
                                                descending=True)[:row_rank]
                row_indices_list.append(torch.sort(cls_row_indices).values)
            self.row_indices = torch.cat(row_indices_list)

        # Compute scale factors
        row_scale_factors = 1 / torch.sqrt(self.stable_rank *
                                           row_probs[self.row_indices])
        col_scale_factors = 1 / torch.sqrt(self.stable_rank *
                                           col_probs[self.column_indices])

        # Create C, R, and W matrices
        C_matrix = matrix[:, self.column_indices] * col_scale_factors
        R_matrix = matrix[self.row_indices, :] * row_scale_factors.unsqueeze(1)
        W_matrix = matrix[self.row_indices, :][:, self.column_indices]

        return C_matrix, W_matrix, R_matrix
