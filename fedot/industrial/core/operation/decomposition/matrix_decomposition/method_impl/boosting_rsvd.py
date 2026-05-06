from typing import Optional

# TODO someone: create save import like for TensorData (data_reader.py)
import cupy as cp
from fedot.core.operations.operation_parameters import OperationParameters
from py_boost.multioutput.sketching import GradSketch

from fedot.industrial.core.operation.decomposition.matrix_decomposition.method_impl.power_iteration_decomposition import \
    RSVDDecomposition


class RandomSVD(GradSketch):
    """TopOutputs sketching. Use only gradient columns with the highest L2 norm"""

    def __init__(self, sample=None, rsvd_params: Optional[OperationParameters] = {}):
        """

        Args:
            sample: int, subsample to speed up SVD fitting
            **svd_params: dict, SVD params, see cuml.TruncatedSVD docs
        """

        self.sample = sample
        self.approximation = rsvd_params.get('approximation', True)
        self.rank = rsvd_params.get('regularized_rank', None)
        self.regularisation = rsvd_params.get('reg_type', 'hard_thresholding')
        self.solver = RSVDDecomposition(rsvd_params)
        self.counter = 0

    def _define_approximation_regime(self, tensor):
        max_num_rows = 10000
        is_matrix_big = any([tensor.shape[0] > max_num_rows, tensor.shape[1] > max_num_rows])
        if is_matrix_big:
            self.approximation = True
        else:
            self.approximation = False

    def _scheduler(self):
        self.counter += 1
        if self.counter % 20 == 0:
            return True
        else:
            return False

    def apply_sketch(self, grad, hess):
        grad_approx = cp.asarray(self.solver.rsvd(tensor=grad.get(), approximation=self.approximation,
                                                  reg_type=self.regularisation, regularized_rank=self.rank,
                                                  return_svd=False,
                                                  sampling_regime='column_sampling').astype('float32'))
        if hess.shape[1] > 1:
            hess = cp.asarray(self.solver.rsvd(tensor=hess.get(), approximation=self.approximation,
                                               reg_type=self.regularisation, regularized_rank=self.rank,
                                               return_svd=False))
            hess_approx = cp.clip(hess, 0.01, None)
        else:
            hess_approx = hess
            # hess_approx = self.solver.random_projection @ hess.get()
            # hess_approx = cp.asarray(hess_approx.astype('float32'))
        self.rank = self.solver.regularized_rank
        return grad_approx, hess_approx

    def __call__(self, grad: cp.ndarray, hess: cp.ndarray):
        self._define_approximation_regime(grad)
        if self._scheduler():
            return self.apply_sketch(grad, hess)
        else:
            return grad, hess
