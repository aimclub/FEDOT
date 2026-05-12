from typing import Tuple, Union, Optional

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from numpy import linalg as LA
from sklearn import preprocessing
from sklearn.random_projection import johnson_lindenstrauss_min_dim

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.repository.constanst_repository import DEFAULT_SVD_SOLVER

RANK_REPRESENTATION = Union[int, float]


class CURDecomposition:
    """
    CUR decomposition is a low-rank matrix decomposition method that is based on selecting
    a subset of columns and rows of the original matrix. The method is based on the
    Johnson-Lindenstrauss lemma and is used to approximate the original matrix with a
    low-rank matrix. The CUR decomposition is defined as follows:
    A = C @ U @ R
    where A is the original matrix, C is a subset of columns of A, U is a subset of rows of A,
    and R is a subset of rows of A. The selection of columns and rows is based on the
    probabilities p and q, which are computed based on the norms of the columns and rows of A.
    The selection of columns and rows is done in such a way that the approximation error is minimized.

    Args:
        params: the parameters of the operation
            rank: the rank of the decomposition
            tolerance: the tolerance of the decomposition
            return_samples: whether to return the samples or the decomposition matrices

    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.stable_rank = params.get('rank', None)
        self.tolerance = params.get('tolerance', [0.5, 0.1, 0.05])
        self.return_samples = params.get('return_samples', True)
        self.column_indices = None
        self.row_indices = None
        self.classes_idx = None
        self.column_space = 'Full'
        self.fitted = True

    def _convert_to_output(self,
                           prediction: np.ndarray,
                           predict_data: InputData,
                           output_data_type: DataTypesEnum = DataTypesEnum.table) -> OutputData:
        """Method convert prediction into :obj:`OutputData` if it is not this type yet

        Args:
            prediction: output from model implementation
            predict_data: :obj:`InputData` used for prediction
            output_data_type: :obj:`DataTypesEnum` for output

        Returns: prediction as :obj:`OutputData`
        """

        if not isinstance(prediction, OutputData):
            # Wrap prediction as OutputData
            converted = OutputData(idx=predict_data.idx,
                                   features=predict_data.features,
                                   predict=prediction,
                                   task=predict_data.task,
                                   target=predict_data.target,
                                   data_type=output_data_type,
                                   supplementary_data=predict_data.supplementary_data)
        else:
            converted = prediction

        return converted

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
        min_num_samples = johnson_lindenstrauss_min_dim(
            n_samples, eps=self.tolerance).tolist()
        self.stable_rank = min(
            [x if x < n_samples else n_samples for x in min_num_samples])
        if isinstance(self.stable_rank, float):
            self.stable_rank = round(max(matrix.shape) * self.stable_rank)

    def get_aproximation_error(self, original_tensor, cur_matrices: tuple):
        C, U, R = cur_matrices
        return np.linalg.norm(original_tensor - C @ U @ R)

    def _balance_target(self, target):
        classes = np.unique(target)
        self.classes_idx = [np.where(target == cls)[0] for cls in classes]

    def fit_transform(self, feature_tensor: np.ndarray,
                      target: np.ndarray = None) -> tuple:
        feature_tensor = feature_tensor.squeeze()
        # transformer = random_projection.SparseRandomProjection().fit_transform(target)
        if self.stable_rank is None:
            self._get_stable_rank(feature_tensor)
        self._balance_target(target)
        # create sub matrices for CUR-decompostion
        array = np.array(feature_tensor.copy())
        c, w, r = self.select_rows_cols(array)
        if self.return_samples:
            sampled_tensor = feature_tensor[:, self.column_indices]
            sampled_tensor = sampled_tensor[self.row_indices, :]
        else:
            # evaluate pseudoinverse for W - U^-1
            X, Sigma, y_T = DEFAULT_SVD_SOLVER(w, full_matrices=False)
            Sigma_plus = np.linalg.pinv(np.diag(Sigma))
            # aprox U using pseudoinverse
            u = y_T.T @ Sigma_plus @ Sigma_plus @ X.T
            sampled_tensor = (c, u, r)
            self.get_aproximation_error(feature_tensor, sampled_tensor)
        if target is not None:
            target = target[self.row_indices]
        self.fitted = True
        return sampled_tensor, target

    def decompose(self, tensor: np.ndarray):
        # transformer = random_projection.SparseRandomProjection().fit_transform(target)
        if self.stable_rank is None:
            self._get_stable_rank(tensor)
        # create sub matrices for CUR-decompostion
        array = np.array(tensor.copy())
        c, w, r = self.select_rows_cols(array)
        # evaluate pseudoinverse for W - U^-1
        U, Sigma, VT = DEFAULT_SVD_SOLVER(w, full_matrices=False)
        Sigma_plus = np.linalg.pinv(np.diag(Sigma))
        # aprox U using pseudoinverse
        u = VT.T @ Sigma_plus @ Sigma_plus @ U.T
        return (c, u, r)

    def transform(self, input_data: InputData) -> tuple:
        if not self.fitted:
            sampled_tensor, samplet_target = self.fit_transform(
                input_data.features, input_data.target)
            output_data = self._convert_to_output(sampled_tensor, input_data)
            output_data.target = samplet_target
        else:
            output_data = self._convert_to_output(
                input_data.features, input_data)
        return output_data

    def reconstruct_basis(self, C, U, R, ts_length):
        # if len(U.shape) > 1:
        #     multi_reconstruction = lambda x: self.reconstruct_basis(C=C, U=U, R=x, ts_length=ts_length)
        #     TS_comps = list(map(multi_reconstruction, R))
        # else:
        rank = U.shape[1]
        TS_comps = np.zeros((ts_length, rank))
        for i in range(rank):
            X_elem = np.outer(C @ U[:, i], R[i, :])
            X_rev = X_elem[::-1]
            eigenvector = [X_rev.diagonal(
                j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]
            TS_comps[:, i] = eigenvector
        return TS_comps

    def select_rows_cols(
            self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Evaluate norms for columns and rows
        matrix = preprocessing.MinMaxScaler().fit_transform(np.nan_to_num(matrix))
        col_norms, row_norms = np.nan_to_num(
            LA.norm(matrix, axis=0)), np.nan_to_num(LA.norm(matrix, axis=1))
        matrix_norm = LA.norm(matrix, 'fro')  # np.sum(np.power(matrix, 2))

        # Compute the probabilities for selecting columns and rows
        col_probs, row_probs = col_norms / matrix_norm, row_norms / matrix_norm
        col_rank, row_rank = self.stable_rank, self.stable_rank
        self.column_indices = np.sort(np.argsort(col_probs)[-col_rank:])
        self.row_indices = np.sort(np.argsort(row_probs)[-col_rank:])
        if self.classes_idx is not None:
            self.row_indices = np.concatenate([np.sort(np.argsort(row_probs[cls_idx])[-row_rank:])
                                               for cls_idx in self.classes_idx])

        row_scale_factors = 1 / \
            np.sqrt(self.stable_rank * row_probs[self.row_indices])
        col_scale_factors = 1 / \
            np.sqrt(self.stable_rank * col_probs[self.column_indices])

        C_matrix = matrix[:, self.column_indices] * col_scale_factors
        R_matrix = matrix[self.row_indices, :] * \
            row_scale_factors[:, np.newaxis]
        W_matrix = matrix[self.row_indices, :][:, self.column_indices]
        # Select k columns and rows based on the probabilities p and q
        # row_probs = preprocessing.Normalizer(norm='l1').fit_transform(row_probs.reshape(1, -1)).flatten()
        # selected_cols = np.random.choice(matrix.shape[1], size=self.rank, replace=False, p=col_probs)
        # selected_rows = np.random.choice(matrix.shape[0], size=row_rank, replace=False, p=row_probs)
        return C_matrix, W_matrix, R_matrix

    @staticmethod
    def ts_to_matrix(time_series: np.ndarray, window: int) -> np.ndarray:
        """Make matrix from ts using window"""

        matrix = np.zeros((len(time_series) - window + 1, window))
        for i in range(len(time_series) - window + 1):
            matrix[i] = time_series[i:i + window]
        return matrix

    @staticmethod
    def matrix_to_ts(matrix: np.ndarray) -> np.ndarray:
        """Make ts from matrix"""

        ts = np.zeros(matrix.shape[0] + matrix.shape[1] - 1)
        for i in range(matrix.shape[0]):
            ts[i:i + matrix.shape[1]] += matrix[i]
        return ts
