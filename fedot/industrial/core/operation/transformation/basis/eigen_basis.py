from typing import Optional

import dask
import tensorly as tl
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from pymonad.either import Either
from pymonad.list import ListMonad
from tensorly.decomposition import parafac
from threadpoolctl import threadpool_limits
from tqdm.dask import TqdmCallback

from fedot.industrial.core.architecture.preprocessing.data_convertor import DataConverter, NumpyConverter
from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.operation.decomposition.matrix_decomposition.decomposer import MatrixDecomposer
from fedot.industrial.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation
from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix
from fedot.industrial.core.operation.transformation.regularization.spectrum import reconstruct_basis, \
    singular_value_hard_threshold


class EigenBasisImplementation(BasisDecompositionImplementation):
    """Eigen Basis decomposition implementation
        Example:
            ts1 = np.random.rand(200)
            ts2 = np.random.rand(200)
            ts = [ts1, ts2]
            bss = EigenBasisImplementation({'window_size': 30})
            basis_multi = bss._transform(ts)
            basis_1d = bss._transform(ts1)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = self.params.get('window_size', 20)
        self.decomposition_type = params.get('decomposition_type', 'svd')
        self.rank_regularization = self.params.get('rank_regularization', 'hard_thresholding')
        self.logging_params.update({'WS': self.window_size})
        self.explained_dispersion = None
        self.SV_threshold = None
        self.solver = MatrixDecomposer({'decomposition_type': self.decomposition_type,
                                        'decomposition_params': {'spectrum_regularization': self.rank_regularization}})

    def __repr__(self):
        return 'EigenBasisImplementation'

    def _tensor_decompose(self, features):
        number_of_dim = list(range(features.shape[1]))
        one_dim_predict = len(number_of_dim) == 1
        predict = []
        if self.SV_threshold is None:
            self.SV_threshold = self.get_threshold(data=features)
            self.logging_params.update({'SV_thr': self.SV_threshold})
        if one_dim_predict:
            evaluation_results = list(map(lambda sample: self._transform_one_sample(sample), features[:, 0, :]))
        else:
            evaluation_results = list(map(lambda dimension: [self._transform_one_sample(sample)
                                                             for sample in features[:, dimension, :]], number_of_dim))
        with TqdmCallback(desc=fr"compute_feature_extraction_with_{self.__repr__()}"):
            with threadpool_limits(limits=1, user_api='blas'):
                feature_matrix = dask.compute(*evaluation_results)

        if not one_dim_predict:
            feature_matrix = np.array(feature_matrix)
            feature_matrix = np.swapaxes(feature_matrix, 0, 1)

        predict = [[np.array(v) if len(v) > 1 else v[0] for v in feature_matrix]]
        return predict

    def _convert_basis_to_predict(self, basis, input_data):

        if input_data.features.shape[0] == 1 and len(
                input_data.features.shape) == 3:
            self.predict = basis[np.newaxis, :, :]
        else:
            self.predict = basis
        if input_data.task.task_params is None:
            input_data.task.task_params = self.__repr__()
        elif input_data.task.task_params not in [self.__repr__(), 'LargeFeatureSpace']:
            input_data.task.task_params.feature_filter = self.__repr__()

        predict = OutputData(idx=input_data.idx,
                             features=input_data.features,
                             predict=self.predict,
                             task=input_data.task,
                             target=input_data.target,
                             data_type=DataTypesEnum.table,
                             supplementary_data=input_data.supplementary_data)
        return predict

    def _transform(self, input_data: InputData) -> np.array:
        """
        Method for transforming all samples
        """
        features = DataConverter(data=input_data).convert_to_monad_data()
        features = NumpyConverter(data=features).convert_to_torch_format()

        basis = np.array(Either.insert(features).then(self._tensor_decompose).value[0])
        predict = self._convert_basis_to_predict(basis, input_data)
        return predict

    def _get_1d_basis(self, data: dict):
        def inverse_transformation(data):
            return reconstruct_basis(U=data['left_eigenvectors'],
                                     Sigma=data['spectrum'],
                                     VT=data['right_eigenvectors'],
                                     ts_length=self.ts_length)

        def threshold(data):
            data['spectrum'] = data['spectrum'][:self.SV_threshold]
            return data

        basis = Either.insert(data).then(threshold).then(inverse_transformation).value
        return np.swapaxes(basis, 1, 0)

    def _get_multidim_basis(self, data):
        rank = round(data.shape[2] / 10)
        beta = data.shape[2] / data.shape[0]

        def tensor_decomposition(x): return ListMonad(
            parafac(tl.tensor(x), rank=rank).factors)

        def multi_threshold(x): return singular_value_hard_threshold(
            singular_values=x, beta=beta, threshold=None)

        def threshold(Monoid): return ListMonad(
            [Monoid[0], list(map(multi_threshold, Monoid[1])), Monoid[2].T])

        def data_driven_basis(Monoid): return ListMonad(reconstruct_basis(
            Monoid[0], Monoid[1], Monoid[2], ts_length=data.shape[2]))

        basis = np.array(Either.insert(data).then(tensor_decomposition).then(
            threshold).then(data_driven_basis).value[0])

        basis = basis.reshape(basis.shape[1], -1)

        return basis

    def get_threshold(self, data) -> int:
        number_of_dim = list(range(data.shape[1]))
        one_dim_predict = len(number_of_dim) == 1

        def mode_func(x):
            return max(set(x), key=x.count)

        if one_dim_predict:
            svd_numbers = list(map(lambda sample:
                                   self._transform_one_sample(sample, svd_flag=True), data[:, 0, :]))
        else:
            dimension_rank = []
            svd_numbers = list(map(lambda dimension:
                                   [dimension_rank.append(self._transform_one_sample(signal, svd_flag=True))
                                    for signal in data[:, dimension, :]], number_of_dim))

            to_comp = []
            for dim in number_of_dim:
                dim_ranks = []
                for sign in data[:, dim, :]:
                    r = self._transform_one_sample(sign, svd_flag=True)
                    dim_ranks.append(r.compute())
                to_comp.append(dim_ranks)

        with threadpool_limits(limits=1, user_api='blas'):
            if not one_dim_predict:
                list_of_ranks = np.array(to_comp).flatten().tolist()
            else:
                list_of_ranks = dask.compute(*svd_numbers)
        common_rank = mode_func(list_of_ranks)
        return common_rank

    @dask.delayed
    def _transform_one_sample(self, series: np.array, svd_flag: bool = False):
        window_size = round(series.shape[0] * (self.window_size / 100))
        trajectory_transformer = HankelMatrix(
            time_series=series, window_size=window_size)
        data = trajectory_transformer.trajectory_matrix
        self.ts_length = trajectory_transformer.ts_length
        basis = Either.insert(data).then(self.solver.apply).value
        if svd_flag:
            return basis['rank']
        else:
            return self._get_1d_basis(basis)
