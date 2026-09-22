from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from scipy.linalg import solve_triangular
from sklearn.decomposition import PCA
# from core.operation.transformation.regularization.lp_reg import compute_penalty_matrix


class FunctionalPCA:
    """
    Principal component analysis.
    Class that implements functional principal component analysis for both
    basis and natural representations of the data.

    Parameters:
        model_hyperparams:
                        n_components: Number of principal components to keep from
                            functional principal component analysis. Defaults to 3.
                        regularization: Regularization object to be applied.
                        basis_of_function: .
    Attributes:
        components\\_: this contains the principal components.
        explained_variance\\_ : The amount of variance explained by
            each of the selected components.
        explained_variance_ratio\\_ : this contains the percentage
            of variance explained by each principal component.
        singular_values\\_: The singular values corresponding to each of the
            selected components.
        mean\\_: mean of the train data.
    Examples:

    time_series = np.array([1,2,3,4,5,6])
    data_range = len(time_series)
    basis = ChebyshevBasis(data_range=data_range, n_components=4).decompose(time_series)
    FPCA = FunctionalPCA(2)
    FPCA = FPCA.fit(basis)

    """

    def __init__(
            self, model_hyperparams: dict = None
    ) -> None:
        self.mean_ = 0
        self.n_components = 2
        self.regularization = None
        self.basis_function = None

        if model_hyperparams is not None:
            self.n_components = model_hyperparams['n_components']
            self.regularization = model_hyperparams['regularization']
            # self._weights = model_hyperparams['weights']
            self.basis_function = model_hyperparams['basis_function']

    def _delete_mean(self, X):
        if not isinstance(X, np.ndarray):
            X = X.values
        mean_centred = np.allclose(X, X - np.mean(X, axis=0))
        if not mean_centred:
            self.mean_ = np.mean(X, axis=0)
            X = X - self.mean_
        return X

    def _fit_basis(
            self,
            X: np.ndarray
    ):
        """
        Compute the first n_components principal components and saves them.

        Args:
            X: The functional data object to be analysed.

        Returns:
            self

        References:
            .. [RS05-8-4-2] Ramsay, J., Silverman, B. W. (2005). Basis function
                expansion of the functions. In *Functional Data Analysis*
                (pp. 161-164). Springer.

        """
        X = self._delete_mean(X)
        if self.basis_function is not None:
            # Compute Gram matrix of basis function
            G = self.basis_function.T.dot(self.basis_function)
            # The matrix that are in charge of changing the computed principal
            # components to target matrix is essentially the inner product
            # of both basis.
            J = np.dot(X, self.basis_function)
        else:
            # If no other basis is specified we use the same basis as the
            X.copy()
            # G = pairwise_distance(X.T)
            G = X.T.dot(X)
            J = G

        self._X_basis = X
        self._j_matrix = J

        # # Apply regularization
        # if self.regularization is not None:
        #     regularization_matrix = compute_penalty_matrix(
        #         basis_iterable=(components_basis,),
        #         regularization_parameter=1,
        #         regularization=self.regularization,
        #     )
        #
        #     G = G + regularization_matrix

        # Diagonalisation of Gram Matrix. G = L*L^T
        l_matrix = np.linalg.cholesky(G)

        # we need L^{-1} for a multiplication, there are two possible ways:
        # using solve to get the multiplication result directly or just invert
        # the matrix. We choose solve because it is faster and more stable.
        # The following matrix is needed: L^{-1}*J^T
        l_inv_j_t = solve_triangular(
            l_matrix,
            np.transpose(J),
            lower=True,
        )

        # the final matrix, C(L-1Jt)t for svd or (L-1Jt)-1CtC(L-1Jt)t for PCA

        final_matrix = X @ np.transpose(l_inv_j_t)

        # initialize the pca module provided by scikit-learn
        pca = PCA(n_components=self.n_components)
        pca.fit(final_matrix)

        # we choose solve to obtain the component coefficients for the
        # same reason: it is faster and more efficient
        component_coefficients = solve_triangular(
            np.transpose(l_matrix),
            np.transpose(pca.components_),
            lower=False,
        )

        self.pca = pca
        self.component_coefficients = component_coefficients
        self.components_ = component_coefficients.T @ X.T

        return self

    def _transform_basis(
            self,
            X: np.array
    ):
        """Compute the n_components first principal components score.
        Args:
            X: The functional data object to be analysed.
        Returns:
            Principal component scores.
        """

        # Compute inner product of our data with the components
        return X @ self._j_matrix @ self.component_coefficients

    def fit(
            self,
            X: np.array
    ):
        """
        Compute the n_components first principal components and saves them.
        Args:
            X: The functional data object to be analysed.
        Returns:
            self
        """

        return self._fit_basis(X)

    def transform(
            self,
            X: np.array
    ):
        """
        Compute the ``n_components`` first principal components scores.
        Args:
            X: The functional data object to be analysed.
        Returns:
            Principal component scores.
        """
        X = self._delete_mean(X)
        return self._transform_basis(X)

    def predict(self, test_features, threshold: float = 0.99):

        if isinstance(test_features, list):
            list_of_projection, list_of_outliers = [], []
            for window_slice in test_features:
                current_projection, current_outlier = self._predict(
                    window_slice.T, threshold)
                list_of_projection.append(current_projection)
                list_of_outliers.append(current_outlier)
            return list_of_projection, list_of_outliers
        else:
            return self._predict(test_features.T, threshold)

    def _predict(self, test_features, threshold: float = 0.90):
        projection = self.transform(test_features)
        recover = self.inverse_transform(projection)
        outlier_idx = quantile_filter(
            input_data=test_features, predicted_data=recover)
        return recover, outlier_idx

    def fit_transform(
            self,
            X: np.array,
    ):
        """
        Compute the n_components first principal components and their scores.
        Args:
            X: The functional data object to be analysed.
        Returns:
            Principal component scores.
        """
        return self.fit(X).transform(X)

    def inverse_transform(
            self,
            pc_scores,
    ):
        """
        Compute the recovery from the fitted principal components scores.
        In other words,
        it maps ``pc_scores``, from the fitted functional PCs' space,
        back to the input functional space.
        ``pc_scores`` might be an array returned by ``transform`` method.
        Args:
            pc_scores: ndarray (n_samples, n_components).
        Returns:

        """

        reconstructed = pc_scores @ self.component_coefficients.T

        return reconstructed + self.mean_
