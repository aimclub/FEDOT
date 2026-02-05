from enum import Enum

from fedot.industrial.core.models.detection.probalistic.kalman import UnscentedKalmanFilter
from fedot.industrial.core.models.detection.subspaces.func_pca import FunctionalPCA
from fedot.industrial.core.models.detection.subspaces.sst import SingularSpectrumTransformation
from fedot.industrial.core.operation.transformation.basis.eigen_basis import EigenBasisImplementation
from fedot.industrial.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot.industrial.core.operation.transformation.basis.wavelet import WaveletBasisImplementation
from fedot.industrial.core.operation.transformation.representation.recurrence.recurrence_extractor import RecurrenceExtractor
from fedot.industrial.core.operation.transformation.representation.statistical.quantile_extractor import QuantileExtractor
from fedot.industrial.core.operation.transformation.representation.topological.topological_extractor import TopologicalExtractor


class BasisTransformations(Enum):
    datadriven = EigenBasisImplementation
    wavelet = WaveletBasisImplementation
    Fourier = FourierBasisImplementation


class FeatureGenerator(Enum):
    quantile = QuantileExtractor
    topological = TopologicalExtractor
    recurrence = RecurrenceExtractor


class MlModel(Enum):
    functional_pca = FunctionalPCA
    kalman_filter = UnscentedKalmanFilter
    sst = SingularSpectrumTransformation


class KernelFeatureGenerator(Enum):
    quantile = [{'feature_generator_type': 'statistical',
                 'feature_hyperparams': {
                     'window_mode': True,
                     'window_size': 5
                 }
                 },
                {'feature_generator_type': 'statistical',
                 'feature_hyperparams': {
                     'window_mode': True,
                     'window_size': 10
                 }
                 },
                {'feature_generator_type': 'statistical',
                 'feature_hyperparams': {
                     'window_mode': True,
                     'window_size': 20
                 }
                 },
                {'feature_generator_type': 'statistical',
                 'feature_hyperparams': {
                     'window_mode': True,
                     'window_size': 30
                 }
                 },
                {'feature_generator_type': 'statistical',
                 'feature_hyperparams': {
                     'window_mode': True,
                     'window_size': 40
                 }
                 }
                ]
    wavelet = [
        {'feature_generator_type': 'wavelet',
         'feature_hyperparams': {'wavelet': "mexh",
                                 'n_components': 2}
         },

        {'feature_generator_type': 'wavelet',
         'feature_hyperparams': {'wavelet': "gaus3",
                                 'n_components': 2}
         },

        {'feature_generator_type': 'wavelet',
         'feature_hyperparams': {'wavelet': "morl",
                                 'n_components': 2}
         }
    ]
    recurrence = []
    topological = []
