from enum import Enum
from itertools import chain

from dask_ml.decomposition import PCA as DaskKernelPCA
from fedot.core.operations.evaluation.operation_implementations.data_operations.decompose import \
    DecomposerClassImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_filters import \
    IsolationForestClassImplementation, IsolationForestRegImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_imbalanced_class import \
    ResampleImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import *
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    ExogDataTransformationImplementation, GaussianFilterImplementation, TsSmoothingImplementation
from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import \
    FedotCatBoostRegressionImplementation, FedotCatBoostClassificationImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.arima import \
    STLForecastARIMAImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.statsmodels import \
    AutoRegImplementation, ExpSmoothingImplementation
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.linear_model import (
    Lasso as SklearnLassoReg,
    LogisticRegression as SklearnLogReg,
    Ridge as SklearnRidgeReg,
    SGDRegressor as SklearnSGD
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor

from fedot.industrial.core.models.classification.freq_domain_clf import FrequencyClassificator
from fedot.industrial.core.models.classification.manifold_domain_clf import ManifoldClassificator
from fedot.industrial.core.models.classification.stat_domain_clf import StatClassificator
from fedot.industrial.core.models.detection.anomaly.algorithms.arima_fault_detector import ARIMAFaultDetector
from fedot.industrial.core.models.detection.anomaly.algorithms.convolutional_autoencoder_detector import \
    ConvolutionalAutoEncoderDetector
from fedot.industrial.core.models.detection.anomaly.algorithms.isolation_forest_detector import IsolationForestDetector
from fedot.industrial.core.models.detection.anomaly.algorithms.lstm_autoencoder_detector import LSTMAutoEncoderDetector
from fedot.industrial.core.models.detection.custom.stat_detector import StatisticalDetector
from fedot.industrial.core.models.detection.probalistic.kalman import UnscentedKalmanFilter
from fedot.industrial.core.models.detection.subspaces.sst import SingularSpectrumTransformation
from fedot.industrial.core.models.nn.network_impl.common_model.dummy_nn import DummyOverComplicatedNeuralNetwork
from fedot.industrial.core.models.nn.network_impl.common_model.inception import InceptionTimeModel
from fedot.industrial.core.models.nn.network_impl.common_model.resnet import ResNetModel
from fedot.industrial.core.models.nn.network_impl.feature_extraction.explainable_convolution_model import XCModel
from fedot.industrial.core.models.nn.network_impl.feature_extraction.mini_rocket import MiniRocketExtractor
from fedot.industrial.core.models.nn.network_impl.forecasting_model.deep_tcn import TCNModel
from fedot.industrial.core.models.nn.network_impl.forecasting_model.deepar import DeepAR
from fedot.industrial.core.models.nn.network_impl.forecasting_model.nbeats import NBeatsModel
from fedot.industrial.core.models.nn.network_impl.forecasting_model.tst import TSTModel
from fedot.industrial.core.models.nn.network_impl.lora_nn import LoraModel
from fedot.industrial.core.models.pdl.pairwise_model import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor
from fedot.industrial.core.models.regression.freq_domain_regressor import FrequencyRegressor
from fedot.industrial.core.models.regression.manifold_domain_regressor import ManifoldRegressor
from fedot.industrial.core.models.regression.stat_domain_regressor import StatRegressor
from fedot.industrial.core.models.ts_forecasting.glm import GLMIndustrial
from fedot.industrial.core.models.ts_forecasting.lagged_strategy.eigen_forecaster import EigenAR
from fedot.industrial.core.models.ts_forecasting.lagged_strategy.lagged_forecaster import LaggedAR
from fedot.industrial.core.models.ts_forecasting.lagged_strategy.topo_forecaster import TopologicalAR
from fedot.industrial.core.operation.filtration.channel_filtration import ChannelCentroidFilter
from fedot.industrial.core.operation.transformation.basis.eigen_basis import EigenBasisImplementation
from fedot.industrial.core.operation.transformation.basis.eigen_basis_torch import EigenBasisImplementationTorch
from fedot.industrial.core.operation.transformation.basis.fourier_basis_torch import FourierBasisImplementationTorch
from fedot.industrial.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot.industrial.core.operation.transformation.basis.wavelet import WaveletBasisImplementation
from fedot.industrial.core.operation.transformation.data.bagging import BaggingEnsemble
from fedot.industrial.core.operation.transformation.representation.manifold.riemann_embeding import RiemannExtractor
from fedot.industrial.core.operation.transformation.representation.recurrence.recurrence_extractor import RecurrenceExtractor
from fedot.industrial.core.operation.transformation.representation.statistical.quantile_extractor import QuantileExtractor
from fedot.industrial.core.operation.transformation.torch_backend.statistical.quantile_extractor import TorchQuantileExtractor
from fedot.industrial.core.operation.transformation.representation.topological.topological_extractor import \
    TopologicalExtractor
from fedot.industrial.core.repository.dask_models import DaskLogisticRegression, DaskRidgeRegression
from fedot.industrial.core.repository.excluded import EXCLUDED_OPERATION_MUTATION, TEMPORARY_EXCLUDED


class AtomizedModel(Enum):
    INDUSTRIAL_CLF_PREPROC_MODEL = {
        # for decomposed tasks
        'class_decompose': DecomposerClassImplementation,
        # for imbalanced data
        'resample': ResampleImplementation,

    }
    SKLEARN_CLF_MODELS = {
        # boosting models (bid datasets)
        'xgboost': GradientBoostingClassifier,
        'catboost': FedotCatBoostClassificationImplementation,
        # solo linear models
        'logit': SklearnLogReg,
        # solo tree models
        'dt': DecisionTreeClassifier,
        # ensemble tree models
        'rf': RandomForestClassifier,
        # solo nn models
        'mlp': MLPClassifier,
        # external models
        'lgbm': LGBMClassifier,
        # for detection
        'one_class_svm': OneClassSVM,
        # pairwise model
        'pdl_clf': PairwiseDifferenceClassifier,
        # custom_model
        'industrial_stat_clf': StatClassificator,
        'industrial_freq_clf': FrequencyClassificator,
        'industrial_manifold_clf': ManifoldClassificator,
    }
    INDUSTRIAL_CLF_AUTOML_MODEL = {'industrial_stat_clf': StatClassificator,
                                   'industrial_freq_clf': FrequencyClassificator,
                                   'industrial_manifold_clf': ManifoldClassificator,
                                   'ensembled_features'
                                   'bagging': BaggingEnsemble
                                   }
    INDUSTRIAL_REG_AUTOML_MODEL = {'industrial_stat_reg': StatRegressor,
                                   'industrial_freq_reg': FrequencyRegressor,
                                   'industrial_manifold_reg': ManifoldRegressor,
                                   'bagging': BaggingEnsemble
                                   }
    FEDOT_PREPROC_MODEL = {
        # data standartization
        'scaling': ScalingImplementation,
        'normalization': NormalizationImplementation,
        # missing data
        'simple_imputation': ImputationImplementation,
        # dimension reduction
        'kernel_pca': KernelPCAImplementation,
        'pca': PCAImplementation,
        # feature generation
        # 'topological_extractor': TopologicalFeaturesImplementation
    }
    INDUSTRIAL_PREPROC_MODEL = {
        # data filtration
        'channel_filtration': ChannelCentroidFilter,
        # data projection onto different basis
        'eigen_basis': EigenBasisImplementation,
        'eigen_basis_torch': EigenBasisImplementationTorch,
        'fourier_basis_torch': FourierBasisImplementationTorch,
        'wavelet_basis': WaveletBasisImplementation,
        'fourier_basis': FourierBasisImplementation,
        # feature extraction algorithm
        'recurrence_extractor': RecurrenceExtractor,
        'quantile_extractor': QuantileExtractor,
        'quantile_extractor_torch': TorchQuantileExtractor,
        'riemann_extractor': RiemannExtractor,
        # feature generation
        # 'topological_extractor': TopologicalFeaturesImplementation,
        'topological_extractor': TopologicalExtractor,
        # nn feature extraction algorithm
        'minirocket_extractor': MiniRocketExtractor,
        # 'chronos_extractor': ChronosExtractor,
        # isolation_forest forest
        'isolation_forest_class': IsolationForestClassImplementation,
        'isolation_forest_reg': IsolationForestRegImplementation,
        # ensemble
        'bagging': BaggingEnsemble
    }

    SKLEARN_REG_MODELS = {
        # boosting models (bid datasets)
        'xgbreg': XGBRegressor,
        'sgdr': SklearnSGD,
        # ensemble tree models (big datasets)
        'treg': ExtraTreesRegressor,
        # solo linear models with regularization
        'ridge': SklearnRidgeReg,
        'lasso': SklearnLassoReg,
        # solo tree models (small datasets)
        'dtreg': DecisionTreeRegressor,
        # external models
        'lgbmreg': LGBMRegressor,
        "catboostreg": FedotCatBoostRegressionImplementation,
        # pairwise model
        'pdl_reg': PairwiseDifferenceRegressor,
        # custom
        'industrial_stat_reg': StatRegressor,
        'industrial_freq_reg': FrequencyRegressor,
        'industrial_manifold_reg': ManifoldRegressor,
    }

    FORECASTING_MODELS = {
        # boosting models (bid datasets)
        'ar': AutoRegImplementation,
        'stl_arima': STLForecastARIMAImplementation,
        'ets': ExpSmoothingImplementation,
        # 'glm': GLMIndustrial,
        'eigen_forecaster': EigenAR,
        'topo_forecaster': TopologicalAR,
        'lagged_forecaster': LaggedAR,
        # variational
        'deepar_model': DeepAR,
        'tcn_model': TCNModel,
        'glm': GLMIndustrial
        # 'locf': RepeatLastValueImplementation
    }

    FORECASTING_PREPROC = {
        # 'lagged': LaggedTransformationImplementation,
        # 'sparse_lagged': SparseLaggedTransformationImplementation,
        'smoothing': TsSmoothingImplementation,
        'gaussian_filter': GaussianFilterImplementation,
        'exog_ts': ExogDataTransformationImplementation
    }

    PRIMARY_FORECASTING_MODELS = [
        'ar',
        # 'deepar_model',
        # 'topo_forecaster',
        'lagged_forecaster',
        'eigen_forecaster'
    ]

    ANOMALY_DETECTION_MODELS = {
        # for detection
        'one_class_svm': OneClassSVM,
        'sst': SingularSpectrumTransformation,
        'unscented_kalman_filter': UnscentedKalmanFilter,
        'stat_detector': StatisticalDetector,
        'arima_detector': ARIMAFaultDetector,
        'iforest_detector': IsolationForestDetector,
        'conv_ae_detector': ConvolutionalAutoEncoderDetector,
        'lstm_ae_detector': LSTMAutoEncoderDetector,
        'channel_filtration': ChannelCentroidFilter,
        'gaussian_filter': GaussianFilterImplementation,
        'smoothing': TsSmoothingImplementation,
        # 'topo_detector': UnscentedKalmanFilter
    }
    NEURAL_MODEL_FOR_CLF = {
        # fundamental models
        'inception_model': InceptionTimeModel,
        'resnet_model': ResNetModel
    }

    NEURAL_MODEL = {
        # fundamental models
        'inception_model': InceptionTimeModel,
        'resnet_model': ResNetModel,
        'nbeats_model': NBeatsModel,
        'tcn_model': TCNModel,
        # transformer models
        'tst_model': TSTModel,
        # explainable models
        'xcm_model': XCModel,
        # variational models
        'deepar_model': DeepAR,
        # detection models
        'conv_ae_detector': ConvolutionalAutoEncoderDetector,
        'lstm_ae_detector': LSTMAutoEncoderDetector,
        # linear_dummy_model
        'dummy': DummyOverComplicatedNeuralNetwork,
        # linear_dummy_model
        'lora_model': LoraModel
    }

    # DASK_MODELS = {'logit': DaskLogReg,
    DASK_MODELS = {'logit': DaskLogisticRegression,
                   'kernel_pca': DaskKernelPCA,
                   #    'kernel_pca': DaskKernelPCA,
                   #    'ridge': DaskLinReg
                   'ridge': DaskRidgeRegression
                   }


def default_industrial_availiable_operation(problem: str = 'regression'):
    operation_dict = {'regression': INDUSTRIAL_REG_AUTOML_MODEL.keys(),
                      'ts_forecasting': FORECASTING_MODELS.keys(),
                      'classification': INDUSTRIAL_CLF_AUTOML_MODEL.keys(),
                      'anomaly_detection': ANOMALY_DETECTION_MODELS.keys(),
                      'classification_tabular': SKLEARN_CLF_MODELS.keys(),
                      'regression_tabular': SKLEARN_CLF_MODELS.keys()}
    available_operations = {'ts_forecasting': [operation_dict[problem],
                                               FORECASTING_PREPROC.keys(),
                                               # SKLEARN_REG_MODELS.keys(),
                                               # INDUSTRIAL_PREPROC_MODEL.keys()
                                               ],
                            'classification': [operation_dict[problem],
                                               NEURAL_MODEL_FOR_CLF.keys(),
                                               INDUSTRIAL_PREPROC_MODEL.keys(),
                                               FEDOT_PREPROC_MODEL.keys()],
                            'regression': [operation_dict[problem],
                                           NEURAL_MODEL.keys(),
                                           INDUSTRIAL_PREPROC_MODEL.keys(),
                                           FEDOT_PREPROC_MODEL.keys()],
                            'anomaly_detection': [operation_dict[problem]],
                            'classification_tabular':
                                [['xgboost', 'logit', 'dt', 'rf', 'mlp', 'lgbm', 'scaling', 'normalization',
                                  'simple_imputation', 'kernel_pca']],
                            'regression_tabular': [operation_dict[problem],
                                                   FEDOT_PREPROC_MODEL.keys()]}

    operations = [list(operation_list)
                  for operation_list in chain(available_operations[problem])]
    operations = list(chain(*operations))
    excluded_operations = list(
        chain(*[list(TEMPORARY_EXCLUDED[x]) for x in TEMPORARY_EXCLUDED.keys()]))
    operations = [x for x in operations if x not in EXCLUDED_OPERATION_MUTATION[problem]
                  and x not in excluded_operations]
    return operations


def overload_model_implementation(list_of_model, backend: str = 'default'):
    overload_list = []
    for model_dict in list_of_model:
        for model_impl in model_dict.keys():
            if model_impl in DASK_MODELS.keys() and backend.__contains__('dask'):
                model_dict[model_impl] = DASK_MODELS[model_impl]
        overload_list.append(model_dict)
    return overload_list


MODELS_WITH_DASK_ALTERNATIVE = [
    AtomizedModel.FEDOT_PREPROC_MODEL.value,
    AtomizedModel.SKLEARN_CLF_MODELS.value,
    AtomizedModel.SKLEARN_REG_MODELS.value
]
DASK_MODELS = AtomizedModel.DASK_MODELS.value
SKLEARN_REG_MODELS = AtomizedModel.SKLEARN_REG_MODELS.value
SKLEARN_CLF_MODELS = AtomizedModel.SKLEARN_CLF_MODELS.value
FEDOT_PREPROC_MODEL = AtomizedModel.FEDOT_PREPROC_MODEL.value
INDUSTRIAL_PREPROC_MODEL = AtomizedModel.INDUSTRIAL_PREPROC_MODEL.value
INDUSTRIAL_CLF_PREPROC_MODEL = AtomizedModel.INDUSTRIAL_CLF_PREPROC_MODEL.value
INDUSTRIAL_REG_AUTOML_MODEL = AtomizedModel.INDUSTRIAL_REG_AUTOML_MODEL.value
INDUSTRIAL_CLF_AUTOML_MODEL = AtomizedModel.INDUSTRIAL_CLF_AUTOML_MODEL.value
ANOMALY_DETECTION_MODELS = AtomizedModel.ANOMALY_DETECTION_MODELS.value
NEURAL_MODEL = AtomizedModel.NEURAL_MODEL.value
NEURAL_MODEL_FOR_CLF = AtomizedModel.NEURAL_MODEL_FOR_CLF.value
FORECASTING_MODELS = AtomizedModel.FORECASTING_MODELS.value
FORECASTING_PREPROC = AtomizedModel.FORECASTING_PREPROC.value
PRIMARY_FORECASTING_MODELS = AtomizedModel.PRIMARY_FORECASTING_MODELS.value
