from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.repository.constanst_repository import DEFAULT_SVD_SOLVER


class SVDDecomposition:
    """
    """

    def __init__(self, params: Optional[OperationParameters] = dict(full_matrices=False)):
        self.decomposition_params = params
        self.sampling_regime = params.get('sampling_regime', 'column_sampling')
        self.sampling_method_dict = dict(column_sampling=self._column_sampling,
                                         row_sampling=self._row_sampling)

    def decompose(self, tensor: np.array) -> list:
        """Block Krylov subspace method for computing the SVD of a matrix with a low computational cost.

        Args:
            tensor: matrix to decompose
            approximation: if True, the matrix approximation will be computed
            regularized_rank: rank of the matrix approximation
            reg_type: type of regularization. 'hard_thresholding' or 'explained_dispersion'

        Returns:
            u, s, vt: decomposition

        """
        # Return classic svd decomposition
        return DEFAULT_SVD_SOLVER(tensor, **self.decomposition_params)

    def compute_approximation(self, tensor, low_rank):
        if self.sampling_regime is not None:
            U, S, VT = self.decompose(tensor)
            tensor_aprox = self.sampling_method_dict[self.sampling_regime](
                U, S, VT, tensor, low_rank)
        return tensor_aprox

    def _column_sampling(self, U, S, V, tensor, low_rank):
        column_basis = S @ V
        top_cols_idx = column_basis.argsort()[-low_rank:][::-1]
        return tensor[:, top_cols_idx]

    def _row_sampling(self, U, S, V, tensor, low_rank):
        row_basis = U @ S
        top_rows_idx = row_basis.argsort()[-low_rank:][::-1]
        return tensor[top_rows_idx, :]
