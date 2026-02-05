from typing import Optional, Tuple

import dask
import pywt
from fedot.core.operations.operation_parameters import OperationParameters
from pymonad.either import Either
from pymonad.list import ListMonad

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation
from fedot.industrial.core.repository.constanst_repository import CONTINUOUS_WAVELETS, DISCRETE_WAVELETS, WAVELET_SCALES


class WaveletBasisImplementation(BasisDecompositionImplementation):
    """Wavelet basis
        Example:
            ts1 = np.random.rand(200)
            ts2 = np.random.rand(200)
            ts = [ts1, ts2]
            bss = WaveletBasisImplementation({'n_components': 2, 'wavelet': 'mexh'})
            basis_multi = bss._transform(ts)
            basis_1d = bss._transform(ts1)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.n_components = params.get('n_components')
        self.wavelet = params.get('wavelet')
        self.use_low_freq = params.get('low_freq', False)
        self.scales = params.get('scale', WAVELET_SCALES)
        self.basis = None
        self.discrete_wavelets = DISCRETE_WAVELETS
        self.continuous_wavelets = CONTINUOUS_WAVELETS
        self.return_feature_vector = params.get('compute_heuristic_representation', False)

    def __repr__(self):
        return 'WaveletBasisImplementation'

    def _compute_heuristic_features(self, input_data):
        wp = pywt.WaveletPacket(data=input_data[None, :], wavelet=self.wavelet,
                                maxlevel=3, axis=1,
                                mode='smooth')

        wpd_approximate_3 = wp['aaa'].data.sum()
        wpd_approximate_2 = wp['aa'].data.sum()
        wpd_approximate_1 = wp['a'].data.sum()
        wpd_detail_3 = wp['ddd'].data.sum()
        wpd_detail_2 = wp['dd'].data.sum()
        wpd_detail_1 = wp['d'].data.sum()
        return np.array([wpd_approximate_3, wpd_approximate_2, wpd_approximate_1]).squeeze(), \
            np.array([wpd_detail_3, wpd_detail_2, wpd_detail_1]).squeeze()

    def _decompose_signal(self, input_data) -> Tuple[np.array, np.array]:
        if self.return_feature_vector:
            return self._compute_heuristic_features(input_data)
        else:
            if self.wavelet in self.discrete_wavelets:
                high_freq, low_freq = pywt.dwt(input_data, self.wavelet, 'smooth')

            else:
                high_freq, low_freq = pywt.cwt(data=input_data,
                                               scales=self.scales,
                                               wavelet=self.wavelet)
                low_freq = high_freq[-1, :]
                high_freq = np.delete(high_freq, (-1), axis=0)
                low_freq = low_freq[np.newaxis, :]
            return high_freq, low_freq

    def _decomposing_level(self) -> int:
        """The level of decomposition of the time series.

        Returns:
            The level of decomposition of the time series.
        """
        return pywt.dwt_max_level(len(self.time_series), self.wavelet)

    @dask.delayed
    def _transform_one_sample(self, series: np.array):
        return self._get_basis(series)

    def _get_1d_basis(self, data) -> np.array:

        def decompose(signal): return ListMonad(self._decompose_signal(signal))

        def threshold(Monoid): return ListMonad([Monoid[0][
                                                 :self.n_components],
                                                 Monoid[1]])

        basis = Either.insert(data).then(decompose).then(threshold).value[0]
        basis = np.concatenate(basis)

        return basis[-1, :] if self.use_low_freq else basis

    def _get_multidim_basis(self, data):
        def decompose(multidim_signal):
            return ListMonad(
                list(map(self._decompose_signal, multidim_signal)))

        def select_level(Monoid):
            high_freq = Monoid[0]
            low_freq = Monoid[1]
            return [high_freq[:self.n_components, :]
                    if len(high_freq.shape) > 1 else high_freq, low_freq]

        def threshold(decomposed_signal):
            return list(
                map(select_level, decomposed_signal))

        basis = Either.insert(data).then(decompose).then(threshold).value
        return np.concatenate([np.concatenate(x) for x in basis])
