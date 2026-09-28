import torch
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.industrial.core.repository.constanst_repository import \
    DEFAULT_SVD_SOLVER_TORCH


class SVDDecompositionTorch:
    """
    Singular Value Decomposition (SVD) implementation using PyTorch.

    This class provides methods for performing SVD on tensors and computing
    low-rank approximations using column or row sampling techniques. It supports
    different sampling regimes for matrix approximation.

    Attributes:
        decomposition_params (dict): Parameters for the SVD decomposition.
        sampling_regime (str): Sampling regime to use for matrix approximation
        ('column_sampling' or 'row_sampling').
        sampling_method_dict (dict): Dictionary mapping sampling regimes to
        their respective methods.
    """

    def __init__(self,
                 params: Optional[OperationParameters] = dict(full_matrices=False)):
        self.decomposition_params = params
        self.sampling_regime = params.get('sampling_regime', 'column_sampling')
        self.sampling_method_dict = dict(column_sampling=self._column_sampling,
                                         row_sampling=self._row_sampling)

    def decompose(self, tensor: torch.Tensor) -> list:
        """
        Perform Singular Value Decomposition (SVD) on the input tensor.

        This method computes the SVD of the input tensor using PyTorch's
        built-in SVD solver.

        Args:
            tensor (torch.Tensor): Input tensor to decompose.

        Returns:
            list: A list containing the SVD decomposition matrices U, S and V^T.
        """
        return DEFAULT_SVD_SOLVER_TORCH(tensor, **self.decomposition_params)

    def compute_approximation(self, tensor, low_rank) -> torch.Tensor:
        """
        Compute a low-rank approximation of the input tensor using SVD and
        sampling.

        This method performs SVD on the input tensor and computes a low-rank
        approximation using the specified sampling regime: column or
        row sampling.

        Args:
            tensor (torch.Tensor): Input tensor to approximate.
            low_rank (int): Rank of the low-rank approximation.

        Returns:
            torch.Tensor: Low-rank approximation of the input tensor.
        """
        if self.sampling_regime is not None:
            U, S, VT = self.decompose(tensor)
            tensor_aprox = (
                self.sampling_method_dict[self.sampling_regime](U,
                                                                S,
                                                                VT,
                                                                tensor,
                                                                low_rank)
            )
        return tensor_aprox

    def _column_sampling(self, U, S, VT, tensor, low_rank) -> torch.Tensor:
        """
        Select top columns based on column basis magnitude.

        This method computes the column basis using the SVD results and selects
        the top columns based on their magnitudes.

        Args:
            U (torch.Tensor): Left singular vectors from SVD.
            S (torch.Tensor): Singular values from SVD.
            VT (torch.Tensor): Right singular vectors from SVD.
            tensor (torch.Tensor): Original tensor.
            low_rank (int): Rank of the low-rank approximation.

        Returns:
            torch.Tensor: Tensor with top columns selected.
        """
        S_mat = torch.diag(S)
        column_basis = S_mat @ VT
        scores = torch.norm(column_basis, dim=0)
        top_cols_idx = torch.argsort(scores, descending=True)[:low_rank]
        return tensor[:, top_cols_idx]

    def _row_sampling(self, U, S, VT, tensor, low_rank) -> torch.Tensor:
        """
        Select top rows based on row basis magnitude.

        This method computes the row basis using the SVD results and selects the
        top rows based on their magnitudes.

        Args:
            U (torch.Tensor): Left singular vectors from SVD.
            S (torch.Tensor): Singular values from SVD.
            VT (torch.Tensor): Right singular vectors from SVD.
            tensor (torch.Tensor): Original tensor.
            low_rank (int): Rank of the low-rank approximation.

        Returns:
            torch.Tensor: Tensor with top rows selected.
        """
        S_mat = torch.diag(S)
        row_basis = U @ S_mat
        scores = torch.norm(row_basis, dim=1)
        top_rows_idx = torch.argsort(scores, descending=True)[:low_rank]
        return tensor[top_rows_idx, :]
