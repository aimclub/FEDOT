"""This module contains classes for computing SVD losses based on the torch Module."""
import torch
from torch import Tensor
from torch.linalg import vector_norm, matrix_norm
from torch.nn.modules import Module


class SVDLoss(Module):
    """Base class for singular value decomposition losses.

    Args:
        factor: The hyperparameter by which the calculated loss function is multiplied
            (default: ``1``).
    """

    def __init__(self, factor: float = 1.) -> None:
        super().__init__()
        self.factor = factor


class OrthogonalLoss(SVDLoss):
    """Orthogonality regularizer for unitary matrices obtained by SVD decomposition.

    Args:
        factor: The hyperparameter by which the calculated loss function is multiplied
            (default: ``1``).
    """

    def __init__(self, factor: float = 1.) -> None:
        super().__init__(factor=factor)

    def forward(self, model: Module) -> Tensor:
        """Calculates orthogonality loss.

        Args:
            model: Optimizable module containing SVD decomposed layers.
        """
        loss = 0
        n = 0
        for name, parameter in model.named_parameters():
            if name.split('.')[-1] == 'U':
                n += 1
                U = parameter
                r = U.size()[1]
                E = torch.eye(r, device=U.device)
                loss += matrix_norm(U.transpose(0, 1) @ U - E) ** 2 / r

            elif name.split('.')[-1] == 'Vh':
                Vh = parameter
                r = Vh.size()[0]
                E = torch.eye(r, device=Vh.device)
                loss += matrix_norm(Vh @ Vh.transpose(0, 1) - E) ** 2 / r
        return self.factor * loss / n


class HoyerLoss(SVDLoss):
    """Hoyer regularizer for matrix with singular values obtained by SVD decomposition.

        Args:
        factor: The hyperparameter by which the calculated loss function is multiplied
            (default: ``1``).
    """

    def __init__(self, factor: float = 1.) -> None:
        super().__init__(factor=factor)

    def forward(self, model: Module) -> Tensor:
        """Calculates Hoyer loss.

        Args:
            model: Optimizable module containing SVD decomposed layers.
        """
        loss = 0
        n = 0
        for name, parameter in model.named_parameters():
            if name.split('.')[-1] == 'S':
                n += 1
                S = parameter
                loss += vector_norm(S, ord=1) / vector_norm(S, ord=2)
        return self.factor * loss / n
