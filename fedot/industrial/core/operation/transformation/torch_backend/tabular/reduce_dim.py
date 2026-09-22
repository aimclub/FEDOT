import torch


class PCA_transformation:
    """Class for PCA transformation.

    Attributes:
        self.n_components: int, number of principle components.
        self.explained_variance: float, explained variance.
        self.mean: float, the mean value of matrix.
        self.components: tensor, matrix for PCA transformation.
        self.fitted: bool, if False, then required to be fitted.
    """

    def __init__(
            self,
            n_components=None,
            explained_variance=0.975):
        super().__init__()
        self.n_components = n_components
        self.explained_variance = explained_variance
        self.components = None
        self.fitted = False

    def fit(self, X: torch.Tensor):
        X = X - X.mean()
        U, S, V = torch.linalg.svd(X, full_matrices=False)
        if self.n_components is None:
            varience = (S ** 2) / (S ** 2).sum()
            cumsum = torch.cumsum(varience, dim=0)
            k = (cumsum < self.explained_variance).sum() + 1
        else:
            k = self.n_components
        self.components = V[:k]
        self.fitted = True
        return self

    def forward(self, X: torch.Tensor):
        if not self.fitted:
            raise ValueError("Model is not fitted. Call method fit() first.")
        X = X - X.mean()
        return X @ self.components.T
