import torch
import torch.nn as nn
from fedot.industrial.core.operation.transformation.torch_backend.tabular.tabular_extractor import PCA_transformation
from fastai.torch_core import Module


class PCA_learning_layer(Module):
    """ Linear layer with PCA weights.

    Attributes:
        self.input_dim: int, input dimension.
        self.explained_variance: float, explained variance for PCA.
        self.freeze_epochs: int, number of freezeed learning epochs.
        self.pca_fitted: bool, if False, then required PCA fitting.
        self.fitted: bool, if True, then the layer is fitted.
        self.W: Tensor, the layer's weights.
        self.epoch: int, current epoch.


    article: https://arxiv.org/html/2501.19114v1#alg1
    """

    def __init__(
            self,
            input_dim: int,
            explained_variance: float = 0.975,
            freeze_epochs: int = 40):
        super().__init__()
        self.input_dim = input_dim
        self.explained_variance = explained_variance
        self.freeze_epochs = freeze_epochs
        self.pca_fitted = False
        self.fitted = False
        self.W = None
        self.register_buffer("mean", torch.zeros(input_dim))
        self.epoch = 0

    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        pca = PCA_transformation(explained_variance=self.explained_variance)
        if not self.pca_fitted:
            pca.fit(X)
        self.W = nn.Parameter(pca.components.clone().T.to(X.device), requires_grad=False)
        self.mean.copy_(X.mean(dim=0))
        self.fitted = True

    def unfreeze_weights(self):
        if self.epoch >= self.freeze_epochs and not self.W.requires_grad:
            self.W = nn.Parameter(self.W.data.clone(), requires_grad=True)

    def step_epoch(self):
        self.epoch += 1
        self.unfreeze_weights()

    def forward(self, X: torch.Tensor):
        X_centered = X - self.mean
        return X_centered @ self.W
