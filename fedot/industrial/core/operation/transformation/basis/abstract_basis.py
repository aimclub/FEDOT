from typing import Optional, Union

import dask
import pandas as pd
import torch
from fedot.core.data.input_data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from pymonad.either import Either
from pymonad.list import ListMonad
from tqdm.dask import TqdmCallback

from fedot.industrial.core.architecture.preprocessing.data_convertor import DataConverter, NumpyConverter
from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation
from fedot.industrial.core.repository.constanst_repository import CPU_NUMBERS, MULTI_ARRAY


class BasisDecompositionImplementation(
        IndustrialCachableOperationImplementation):
    """
    A class for decomposing data on the abstract basis and evaluating the derivative of the resulting decomposition.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.n_processes = CPU_NUMBERS
        self.n_components = params.get('n_components', 2)
        self.basis = None
        self.data_type = MULTI_ARRAY
        self.min_rank = 1

        self.logging_params = {'jobs': self.n_processes}

    def _get_basis(self, data):

        if isinstance(data, list) or all(
                [(isinstance(data, np.ndarray) | isinstance(data, torch.Tensor))
                 and len(data.shape) > 1]):
            func = self._get_multidim_basis
        else:
            func = self._get_1d_basis
        basis = Either.insert(data).then(func).value
        return basis

    def fit(self, data):
        """Decomposes the given data on the chosen basis.

        Returns:
            np.array: The decomposition of the given data.
        """

    def _decompose_signal(self, signal) -> np.array:
        pass

    def evaluate_derivative(self, order: int = 1):
        """Evaluates the derivative of the decomposition of the given data.

        Returns:
            np.array: The derivative of the decomposition of the given data.
        """

    def _transform_one_sample(self, sample: np.array):
        """
            Method for transforming one sample
        """

    def _get_1d_basis(self, input_data):
        def decompose(signal): return ListMonad(self._decompose_signal(signal))
        basis = Either.insert(input_data).then(decompose).value[0]
        return basis

    def _transform(self,
                   input_data: Union[InputData,
                                     pd.DataFrame]) -> np.array:
        """Method for transforming all samples

        """
        features = DataConverter(data=input_data).convert_to_monad_data()
        evaluation_results = list(map(lambda sample: self._transform_one_sample(sample), features))
        with TqdmCallback(desc=f"compute_transformation_to_{self.__repr__()}"):
            evaluation_results = dask.compute(*evaluation_results)
        predict = NumpyConverter(data=np.array(evaluation_results)).convert_to_torch_format()
        return predict

    def _get_multidim_basis(self, input_data):
        def decompose(multidim_signal): return ListMonad(
            list(map(self._decompose_signal, multidim_signal)))
        basis = Either.insert(input_data).then(decompose).value[0]
        return basis
