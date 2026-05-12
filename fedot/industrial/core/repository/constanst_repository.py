import math
import pathlib
from enum import Enum
from multiprocessing import cpu_count

import numpy as np
import pywt
import spectrum
import torch
from MKLpy.algorithms import FHeuristic, RMKL, MEMO, CKA, PWMK
from dask_ml.decomposition import TruncatedSVD as DaskSVD
from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import \
    FedotCatBoostRegressionImplementation, FedotCatBoostClassificationImplementation
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from golem.core.tuning.optuna_tuner import OptunaTuner
from golem.core.tuning.sequential import SequentialTuner
from golem.core.tuning.simultaneous import SimultaneousTuner
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from scipy.spatial.distance import euclidean, cosine, cityblock, correlation, chebyshev, \
    minkowski
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.linear_model import (LinearRegression as linreg,
                                  Lasso as SklearnLassoReg,
                                  LogisticRegression as SklearnLogReg,
                                  Ridge as SklearnRidgeReg,
                                  SGDRegressor as SklearnSGD
                                  )
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch import nn
from xgboost import XGBRegressor

from fedot.industrial.core.metrics.metrics_implementation import calculate_classification_metric, calculate_regression_metric, \
    calculate_forecasting_metric, calculate_detection_metric
from fedot.industrial.core.models.nn.network_modules.losses import CenterLoss, CenterPlusLoss, ExpWeightedLoss, FocalLoss, \
    HuberLoss, LogCoshLoss, MaskedLossWrapper, RMSELoss, SMAPELoss, TweedieLoss
from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix
from fedot.industrial.core.operation.transformation.representation.statistical.stat_features import autocorrelation, ben_corr, \
    crest_factor, energy, \
    hjorth_complexity, hjorth_mobility, hurst_exponent, interquartile_range, kurtosis, mean_ema, mean_moving_median, \
    mean_ptp_distance, n_peaks, pfd, ptp_amp, q25, q5, q75, q95, shannon_entropy, skewness, slope, zero_crossing_rate
from fedot.industrial.core.operation.transformation.torch_backend.statistical.stat_features import mean_torch, median_torch, max_torch, min_torch, \
    autocorrelation_torch, ben_corr_torch, std_torch, \
    crest_factor_torch, energy_torch, \
    hjorth_complexity_torch, hjorth_mobility_torch, hurst_exponent_torch, interquantile_range_torch, kurtosis_torch, mean_ema_torch, mean_moving_median_torch, \
    mean_ptp_distance_torch, n_peaks_torch, pfd_torch, ptp_amp_torch, q5_torch, q25_torch, q75_torch, q95_torch, shannon_entropy_torch, skewness_torch, slope_torch, zero_crossing_rate_torch
from fedot.industrial.core.operation.transformation.torch_specter.eigen import pev_torch
from fedot.industrial.core.operation.transformation.torch_specter.speriodogram import speriodogram_torch
from fedot.industrial.core.operation.transformation.representation.topological.topofeatures import AverageHoleLifetimeFeature, \
    AveragePersistenceLandscapeFeature, BettiNumbersSumFeature, HolesNumberFeature, MaxHoleLifeTimeFeature, \
    PersistenceDiagramsExtractor, PersistenceEntropyFeature, RadiusAtMaxBNFeature, RelevantHolesNumber, \
    SimultaneousAliveHolesFeature, SumHoleLifetimeFeature
from fedot.industrial.tools.serialisation.path_lib import PROJECT_PATH

industrial_model_params_dict = dict(quantile_extractor={'window_size': 10,
                                                        'stride': 1,
                                                        'add_global_features': True},
                                    quantile_extractor_torch={'window_size': 10,
                                                              'stride': 1,
                                                              'add_global_features': True},
                                    wavelet_basis={'n_components': 3,
                                                   'wavelet': 'mexh'},
                                    fourier_basis={'low_rank': 5,
                                                   'output_format': 'signal',
                                                   'approximation': 'exact',
                                                   'threshold': 0.9},
                                    eigen_basis={'window_size': 20,
                                                 'rank_regularization': 'explained_dispersion',
                                                 'low_rank_approximation': False,
                                                 'tensor_approximation': False},
                                    ar={'trend': 't',
                                        'seasonal': False},
                                    rf={'random_state': 42,
                                        'n_jobs': -1,
                                        'criterion': 'entropy',
                                        'n_estimators': 300,
                                        'max_depth': 5},
                                    ar_periodic={},
                                    catboost={"allow_writing_files": False,
                                              "verbose": False,
                                              "max_depth": 5,
                                              "learning_rate": 0.1,
                                              "min_data_in_leaf": 3,
                                              "max_bin": 32,
                                              "l2_leaf_reg": 1})


def beta_thr(beta):
    return 0.56 * np.power(beta, 3) - 0.95 * \
        np.power(beta, 2) + 1.82 * beta + 1.43


def get_default_industrial_model_params(model_name):
    return industrial_model_params_dict[model_name]


stat_params = get_default_industrial_model_params('quantile_extractor_torch')
wavelet_params = get_default_industrial_model_params('wavelet_basis')
fourier_params = get_default_industrial_model_params('fourier_basis')
eigen_params = get_default_industrial_model_params('eigen_basis')
ar_params = get_default_industrial_model_params('ar')
rf_params = get_default_industrial_model_params('rf')
catboost_params = get_default_industrial_model_params('catboost')


class ComputationalConstant(Enum):
    CPU_NUMBERS = math.ceil(cpu_count() * 0.7) if cpu_count() > 1 else 1
    GLOBAL_IMPORTS = {
        'numpy': 'np',
        'cupy': 'np',
        'torch': 'torch',
        'torch.nn': 'nn',
        'torch.nn.functional': 'F'
    }
    BATCH_SIZE_FOR_FEDOT_WORKER = 1000
    FEDOT_WORKER_NUM = 5
    FEDOT_WORKER_TIMEOUT_PARTITION = 4
    PATIENCE_FOR_EARLY_STOP = 15


class KernelsConstant(Enum):
    KERNEL_ALGO = {
        'one_step_heur': FHeuristic,
        'two_step_rmkl': RMKL,
        'two_step_memo': MEMO,
        'one_step_cka': CKA,
        'one_step_pwmk': PWMK,
    }
    KERNEL_BASELINE_FEATURE_GENERATORS = {
        # 'minirocket_extractor': PipelineBuilder().add_node('minirocket_extractor'),
        # 'topological_extractor': PipelineBuilder().add_node('topological_extractor'),
        'quantile_extractor': PipelineBuilder().add_node('quantile_extractor', params=stat_params),
        'wavelet_extractor': PipelineBuilder().add_node('wavelet_basis', params=wavelet_params).
        add_node('quantile_extractor', params=stat_params),
        'fourier_extractor': PipelineBuilder().add_node('fourier_basis', params=fourier_params).
        add_node('quantile_extractor', params=stat_params),
        'eigen_extractor': PipelineBuilder().add_node('eigen_basis', params=eigen_params).
        add_node('quantile_extractor', params=stat_params), }

    KERNEL_BASELINE_FEATURE_GENERATORS_TORCH = {
        'quantile_extractor': PipelineBuilder().add_node('quantile_extractor_torch', params=stat_params),
        'fourier_extractor': PipelineBuilder().add_node('fourier_basis_torch', params=fourier_params).
        add_node('quantile_extractor_torch', params=stat_params),
        'eigen_extractor': PipelineBuilder().add_node('eigen_basis_torch', params=eigen_params).
        add_node('quantile_extractor_torch', params=stat_params), }

    KERNEL_BASELINE_NODE_LIST = {
        'quantile_extractor': (None, 'quantile_extractor'),
        'topological_extractor': (None, 'topological_extractor'),
        'wavelet_extractor': ('wavelet_basis', 'quantile_extractor'),
        'fourier_extractor': ('fourier_basis', 'quantile_extractor'),
        'eigen_extractor': ('eigen_basis', 'quantile_extractor')}

    KERNEL_DISTANCE_METRIC = {'l_metric': [
        'chebyshev',  # L_inf (max distance by coord)
        'cityblock',  # L1 metric
        'correlation',  # pearson correlation
        'cosine',  # cosine distance
        'minkowski',  # L metric
    ],
        'boolean_metric': ['braycurtis',  # for categorical/binary data
                           'hamming',  # for categorical/binary data
                           'dice',  # for categorical/binary data
                           'canberra',  # weighted L1 metric (for ranked data)
                           'dice',  # for categorical/binary data
                           'rogerstanimoto',  # for categorical/binary data
                           'russellrao',  # for categorical/binary data
                           'sokalmichener', ],
        'probability_metric': ['jensenshannon',  # for probability vectors/matrix
                               'mahalanobis',  # for probability vectors/matrix
                               ],
        'default_metric': 'cosine'
    }


class DataTypeConstant(Enum):
    MULTI_ARRAY = DataTypesEnum.image
    MATRIX = DataTypesEnum.table
    TRAJECTORY_MATRIX = HankelMatrix


class PathConstant(Enum):
    IND_DATA_OPERATION_PATH = pathlib.Path(PROJECT_PATH, 'fedot', 'industrial', 'core', 'repository', 'data',
                                           'industrial_data_operation_repository.json')
    DEFAULT_DATA_OPERATION_PATH = pathlib.Path(
        'data_operation_repository.json')
    IND_MODEL_OPERATION_PATH = pathlib.Path(PROJECT_PATH, 'fedot', 'industrial', 'core', 'repository', 'data',
                                            'industrial_model_repository.json')
    DEFAULT_MODEL_OPERATION_PATH = pathlib.Path('model_repository.json')


class SolverConstant(Enum):
    SOLVER_MODELS = {'np_svd_solver': np.linalg.svd,
                     'np_qr_solver': np.linalg.qr,
                     'dask_svd_solver': DaskSVD,
                     'torch_svd_solver': torch.linalg.svd,
                     'torch_qr_solver': torch.linalg.qr,
                     }


class FeatureConstant(Enum):
    STAT_METHODS = {
        'mean_': np.mean,
        'median_': np.median,
        'std_': np.std,
        'max_': np.max,
        'min_': np.min,
        'q5_': q5,
        'q25_': q25,
        'q75_': q75,
        'q95_': q95
    }
    STAT_METHODS_TORCH = {
        'mean_': mean_torch,
        'median_': median_torch,
        'std_': std_torch,
        'max_': max_torch,
        'min_': min_torch,
        'q5_': q5_torch,
        'q25_': q25_torch,
        'q75_': q75_torch,
        'q95_': q95_torch
    }

    BAGGING_METHOD = {
        'mean': np.mean,
        'median': np.median,
        'max': np.max,
        'min': np.min,
        'weighted': linreg
    }

    STAT_METHODS_GLOBAL = {
        'skewness_': skewness,
        'kurtosis_': kurtosis,
        'n_peaks_': n_peaks,
        'slope_': slope,
        'ben_corr_': ben_corr,
        'interquartile_range_': interquartile_range,
        'energy_': energy,
        'cross_rate_': zero_crossing_rate,
        'autocorrelation_': autocorrelation,
        'shannon_entropy_': shannon_entropy,
        'ptp_amplitude_': ptp_amp,
        'mean_ptp_distance_': mean_ptp_distance,
        'crest_factor_': crest_factor,
        'mean_ema_': mean_ema,
        'mean_moving_median_': mean_moving_median,
        'hjorth_mobility_': hjorth_mobility,
        'hjorth_complexity_': hjorth_complexity,
        'hurst_exponent_': hurst_exponent,
        'petrosian_fractal_dimension_': pfd
    }

    STAT_METHODS_GLOBAL_TORCH = {
        'skewness_': skewness_torch,
        'kurtosis_': kurtosis_torch,
        'n_peaks_': n_peaks_torch,
        'slope_': slope_torch,
        'ben_corr_': ben_corr_torch,
        'interquartile_range_': interquantile_range_torch,
        'energy_': energy_torch,
        'cross_rate_': zero_crossing_rate_torch,
        'autocorrelation_': autocorrelation_torch,
        'shannon_entropy_': shannon_entropy_torch,
        'ptp_amplitude_': ptp_amp_torch,
        'mean_ptp_distance_': mean_ptp_distance_torch,
        'crest_factor_': crest_factor_torch,
        'mean_ema_': mean_ema_torch,
        'mean_moving_median_': mean_moving_median_torch,
        'hjorth_mobility_': hjorth_mobility_torch,
        'hjorth_complexity_': hjorth_complexity_torch,
        'hurst_exponent_': hurst_exponent_torch,
        'petrosian_fractal_dimension_': pfd_torch
    }

    METRICS_DICT = {'euclidean': euclidean,
                    'cosine': cosine,
                    'cityblock': cityblock,
                    'correlation': correlation,
                    'chebyshev': chebyshev,
                    # 'jensenshannon': jensenshannon,
                    # 'mahalanobis': mahalanobis,
                    'minkowski': minkowski
                    }

    SPECTRUM_ESTIMATORS = dict(ma=spectrum.ma,
                               non_parametric=spectrum.Periodogram,
                               parma=spectrum.parma,
                               yule=spectrum.pyule,
                               burg=spectrum.pburg,
                               covar=spectrum.pcovar,
                               minvar=spectrum.pminvar,
                               eigen=spectrum.pev,
                               )

    SPECTRUM_ESTIMATORS_TORCH = dict(non_parametric=speriodogram_torch,
                                     eigen=pev_torch,
                                     )

    DEFAULT_ESTIMATOR_PARAMETERS = dict(
        non_parametric=dict(
            window="hann",
            detrend=False,
            scale_by_freq=False,
            NFFT=None,
        ),
        eigen=dict(IP=5,
                   NFFT=None,
                   ),
    )

    PERSISTENCE_DIAGRAM_FEATURES = {
        'HolesNumberFeature': HolesNumberFeature(),
        'MaxHoleLifeTimeFeature': MaxHoleLifeTimeFeature(),
        'RelevantHolesNumber': RelevantHolesNumber(),
        'AverageHoleLifetimeFeature': AverageHoleLifetimeFeature(),
        'SumHoleLifetimeFeature': SumHoleLifetimeFeature(),
        'PersistenceEntropyFeature': PersistenceEntropyFeature(),
        'SimultaneousAliveHolesFeature': SimultaneousAliveHolesFeature(),
        'AveragePersistenceLandscapeFeature': AveragePersistenceLandscapeFeature(),
        'BettiNumbersSumFeature': BettiNumbersSumFeature(),
        'RadiusAtMaxBNFeature': RadiusAtMaxBNFeature()}

    PERSISTENCE_DIAGRAM_EXTRACTOR = PersistenceDiagramsExtractor(
        takens_embedding_dim=1, takens_embedding_delay=2, homology_dimensions=(
            0, 1), parallel=False)
    DISCRETE_WAVELETS = pywt.wavelist(kind='discrete')
    CONTINUOUS_WAVELETS = pywt.wavelist(kind='continuous')
    WAVELET_SCALES = [2, 4, 10, 20]
    SINGULAR_VALUE_MEDIAN_THR = 2.58
    SINGULAR_VALUE_BETA_THR = beta_thr


class FedotOperationConstant(Enum):
    EXCLUDED_OPERATION = ['fast_ica']
    FEDOT_TASK = {
        'classification': Task(
            TaskTypesEnum.classification), 'regression': Task(
            TaskTypesEnum.regression), 'ts_forecasting': Task(
            TaskTypesEnum.ts_forecasting, TsForecastingParams(
                forecast_length=1)), 'anomaly_detection': Task(
            TaskTypesEnum.classification)}

    FEDOT_API_PARAMS = default_param_values_dict = dict(
        problem=None,
        task_params=None,
        timeout=None,
        n_jobs=-1,
        logging_level=50,
        seed=42,
        parallelization_mode='populational',
        show_progress=True,
        max_depth=6,
        max_arity=3,
        pop_size=20,
        num_of_generations=None,
        keep_n_best=1,
        available_operations=None,
        metric=None,
        cv_folds=2,
        genetic_scheme=None,
        early_stopping_iterations=None,
        early_stopping_timeout=10,
        optimizer=None,
        collect_intermediate_metric=False,
        max_pipeline_fit_time=None,
        initial_assumption=None,
        preset=None,
        use_pipelines_cache=True,
        use_preprocessing_cache=True,
        use_input_preprocessing=True,
        use_auto_preprocessing=False,
        use_meta_rules=False,
        cache_dir=None,
        keep_history=True,
        history_dir=None,
        with_tuning=True)
    FEDOT_GET_METRICS = {'regression': calculate_regression_metric,
                         'ts_forecasting': calculate_forecasting_metric,
                         'classification': calculate_classification_metric,
                         'anomaly_detection': calculate_detection_metric
                         }
    FEDOT_TUNING_METRICS = {
        'classification': ClassificationMetricsEnum.f1,
        'ts_forecasting': RegressionMetricsEnum.RMSE,  # RegressionMetricsEnum.MAPE,
        'regression': RegressionMetricsEnum.RMSE}
    FEDOT_DATA_TYPE = {
        'tensor': DataTypesEnum.image,
        'time_series': DataTypesEnum.ts,
        'table': DataTypesEnum.table}
    FEDOT_TUNER_STRATEGY = {
        'optuna': OptunaTuner,
        'simultaneous': SimultaneousTuner,
        'sequential': SequentialTuner,
    }
    FEDOT_HEAD_ENSEMBLE = {'regression': 'treg',
                           'classification': 'xgboost'}
    FEDOT_ATOMIZE_OPERATION = {'regression': 'fedot_regr',
                               'classification': 'fedot_cls'}
    AVAILABLE_CLS_OPERATIONS = [
        'rf',
        'logit',
        'scaling',
        'normalization',
        'xgboost',
        'dt',
        'mlp',
        'kernel_pca']

    AVAILABLE_ANOMALY_DETECTION_OPERATIONS = [
        'sst',
        'unscented_kalman_filter',
        'channel_filtration',
        'gaussian_filter',
        'smoothing'
    ]

    AVAILABLE_REG_OPERATIONS = [
        'scaling',
        'normalization',
        'xgbreg',
        'dtreg',
        'treg',
        'kernel_pca'
    ]

    FEDOT_ASSUMPTIONS = {
        'classification': PipelineBuilder().
        add_node('quantile_extractor_torch', params=stat_params).add_node(
            'catboost', params=catboost_params),
        'classification_tabular': PipelineBuilder().add_node('rf', params=rf_params),
        'regression': PipelineBuilder().add_node('quantile_extractor_torch', params=stat_params).add_node('treg'),
        'regression_tabular': PipelineBuilder().add_node('treg'),
        'anomaly_detection': PipelineBuilder().add_node('iforest_detector'),
        'ts_forecasting': PipelineBuilder().add_node('ar')
    }

    FEDOT_TS_FORECASTING_ASSUMPTIONS = {
        'nbeats': PipelineBuilder().add_node('nbeats_model'),
    }

    FEDOT_INDUSTRIAL_STRATEGY = ['federated_automl',
                                 'kernel_automl',
                                 'forecasting_assumptions',
                                 'forecasting_exogenous',
                                 'lora_strategy',
                                 'sampling_strategy']

    FEDOT_ENSEMBLE_ASSUMPTIONS = {
        'classification': PipelineBuilder().add_node('logit'),
        'regression': PipelineBuilder().add_node('treg')
    }
    # mutation order - [param_change,model_change,add_preproc_model,drop_model,add_model]
    FEDOT_MUTATION_STRATEGY = {
        'params_mutation_strategy': [0.7, 0.3, 0.00, 0.00, 0.0],
        'growth_mutation_strategy': [0.15, 0.15, 0.3, 0.1, 0.3],
        'regularization_mutation_strategy': [0.2, 0.3, 0.1, 0.3, 0.1],
        'initial_population_diversity_strategy': [0.0, 0.5, 0.5, 0.0, 0.0],
        'unique_population_strategy': [0.0, 0.25, 0.5, 0.0, 0.25],
    }

    EXPLAINABLE_MODELS = ['recurrence_extractor',
                          ]
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
        "catboostreg": FedotCatBoostRegressionImplementation
    }


class ModelCompressionConstant(Enum):
    ENERGY_THR = [0.9, 0.95, 0.99, 0.999]
    DECOMPOSE_MODE = 'channel'
    FORWARD_MODE = 'one_layer'
    HOER_LOSS = 0.1
    ORTOGONAL_LOSS = 10
    MODELS_FROM_LENGTH = {
        122: 'ResNet18',
        218: 'ResNet34',
        320: 'ResNet50',
        626: 'ResNet101',
        932: 'ResNet152',
    }


class TorchLossesConstant(Enum):
    CROSS_ENTROPY = nn.CrossEntropyLoss
    MULTI_CLASS_CROSS_ENTROPY = nn.BCEWithLogitsLoss
    MSE = nn.MSELoss
    RMSE = RMSELoss
    SMAPE = SMAPELoss
    TWEEDIE_LOSS = TweedieLoss
    FOCAL_LOSS = FocalLoss
    CENTER_PLUS_LOSS = CenterPlusLoss
    CENTER_LOSS = CenterLoss
    MASK_LOSS = MaskedLossWrapper
    LOG_COSH_LOSS = LogCoshLoss
    HUBER_LOSS = HuberLoss
    EXPONENTIAL_WEIGHTED_LOSS = ExpWeightedLoss


class BenchmarkDatasets(Enum):
    MULTI_REG_BENCH = [
        "AppliancesEnergy",
        "AustraliaRainfall",
        "BeijingPM10Quality",
        "BeijingPM25Quality",
        "BenzeneConcentration",
        "BIDMC32HR",
        "BIDMC32RR",
        "BIDMC32SpO2",
        "Covid3Month",
        "FloodModeling1",
        "FloodModeling2",
        "FloodModeling3",
        "HouseholdPowerConsumption1",
        "HouseholdPowerConsumption2",
        "IEEEPPG",
        "LiveFuelMoistureContent",
        "NewsHeadlineSentiment",
        "NewsTitleSentiment",
        "PPGDalia",
    ]
    M4_FORECASTING_BENCH = ['D1002', 'D1019', 'D1032', 'D1091', 'D1101', 'D1104', 'D1124', 'D1162', 'D1170', 'D1204',
                            'D1219', 'D1232', 'D1263',
                            'D1270', 'D1279', 'D1317', 'D1322', 'D1329', 'D133', 'D1368', 'D1378', 'D137', 'D1380',
                            'D1384', 'D1411', 'D1424',
                            'D1432', 'D1433', 'D1434', 'D1438', 'D1448', 'D1450', 'D1459', 'D145', 'D1464', 'D1475',
                            'D1519', 'D152', 'D1540',
                            'D1551', 'D1577', 'D1616', 'D1651', 'D1679', 'D1693', 'D1706', 'D1712', 'D1735', 'D1740',
                            'D1847', 'D1872', 'D1875',
                            'D1949', 'D1964', 'D1972', 'D1988', 'D2013', 'D2025', 'D2027', 'D2032', 'D2057', 'D2088',
                            'D2114', 'D2144', 'D2162',
                            'D2214', 'D2221', 'D2262', 'D2268', 'D2334', 'D2343', 'D237', 'D2393', 'D2484', 'D2486',
                            'D250', 'D2534', 'D2552',
                            'D2600', 'D261', 'D2648', 'D2670', 'D2698', 'D2717', 'D2729', 'D2797', 'D282', 'D2850',
                            'D2878', 'D2904', 'D292',
                            'D2979', 'D2991', 'D3006', 'D3012', 'D3023', 'D3029', 'D3043', 'D3046', 'D3050', 'D3055',
                            'D3067', 'D3082', 'D3083',
                            'D3085', 'D3094', 'D312', 'D3135', 'D3140', 'D3167', 'D3178', 'D319', 'D3222', 'D324',
                            'D3257', 'D3266', 'D3278',
                            'D3293', 'D3298', 'D3302', 'D3322', 'D3337', 'D3353', 'D3364', 'D3386', 'D3481', 'D3498',
                            'D3505', 'D3520', 'D3530',
                            'D3548', 'D3569', 'D3581', 'D3584', 'D3587', 'D3594', 'D3599', 'D3608', 'D3611', 'D3626',
                            'D3637', 'D3678', 'D3681',
                            'D3686', 'D3691', 'D3697', 'D3721', 'D3737', 'D3745', 'D376', 'D3779', 'D3787', 'D3792',
                            'D3816', 'D3819', 'D3823',
                            'D3845', 'D3920', 'D3935', 'D3991', 'D4043', 'D404', 'D4066', 'D4097', 'D4146', 'D4150',
                            'D4167', 'D420', 'D4210',
                            'D4218', 'D435', 'D462', 'D465', 'D535', 'D543', 'D548', 'D569', 'D56', 'D63', 'D640',
                            'D652', 'D678', 'D723',
                            'D750', 'D761', 'D771', 'D825', 'D826', 'D828', 'D852', 'D856', 'D870', 'D8', 'D902',
                            'D910', 'D939', 'D954',
                            'D967', 'D970', 'D973', 'M10641', 'M1080', 'M11010', 'M11230', 'M11654', 'M11779', 'M11806',
                            'M12209', 'M12241',
                            'M12422', 'M12452', 'M13030', 'M13157', 'M13617', 'M13727', 'M13796', 'M14148', 'M15150',
                            'M15443', 'M15510', 'M15559',
                            'M15902', 'M16100', 'M16137', 'M1630', 'M16405', 'M16731', 'M16863', 'M16927', 'M17178',
                            'M17510', 'M17665', 'M17901',
                            'M17917', 'M18021', 'M18132', 'M18267', 'M18489', 'M18550', 'M18663', 'M19181', 'M19216',
                            'M19553', 'M20010', 'M20194',
                            'M20218', 'M20318', 'M20429', 'M2061', 'M20788', 'M2101', 'M21476', 'M22137', 'M22184',
                            'M22916', 'M23005', 'M23136',
                            'M23572', 'M23842', 'M23923', 'M24186', 'M24396', 'M24648', 'M24684', 'M24777', 'M24988',
                            'M25052', 'M25586', 'M25661',
                            'M26386', 'M26416', 'M26418', 'M26506', 'M27047', 'M27900', 'M28161', 'M2830', 'M28678',
                            'M29276', 'M29608', 'M30319',
                            'M30428', 'M30459', 'M30623', 'M30696', 'M3075', 'M31073', 'M31163', 'M31375', 'M31442',
                            'M31954', 'M31964', 'M32473',
                            'M32515', 'M32607', 'M3276', 'M32905', 'M33076', 'M33543', 'M34126', 'M34214', 'M34318',
                            'M34347', 'M34499', 'M34536',
                            'M34564', 'M35054', 'M35216', 'M35305', 'M35407', 'M35499', 'M35872', 'M35933', 'M36401',
                            'M36641', 'M36658', 'M36695',
                            'M36829', 'M36886', 'M37206', 'M37276', 'M37366', 'M37635', 'M37647', 'M37980', 'M38216',
                            'M38297', 'M38361', 'M38371',
                            'M384', 'M38662', 'M3876', 'M39015', 'M3909', 'M39253', 'M3937', 'M39415', 'M3953',
                            'M39588', 'M39702', 'M39715',
                            'M39723', 'M39793', 'M40077', 'M40467', 'M40808', 'M41148', 'M41149', 'M41150', 'M41172',
                            'M41657', 'M41691', 'M42102',
                            'M42327', 'M42494', 'M42518', 'M42862', 'M43022', 'M43330', 'M4334', 'M43394', 'M43760',
                            'M43904', 'M43912', 'M44515',
                            'M44608', 'M45146', 'M45468', 'M45633', 'M45668', 'M46411', 'M47220', 'M47277', 'M47556',
                            'M4845', 'M5151', 'M5216',
                            'M5684', 'M5831', 'M5992', 'M6032', 'M6047', 'M6175', 'M6225', 'M6295', 'M6462', 'M6539',
                            'M6564', 'M6612', 'M6694',
                            'M690', 'M69', 'M8051', 'M8152', 'M8178', 'M8497', 'M8660', 'M9322', 'M9385', 'M9795',
                            'Q10070', 'Q10262', 'Q10292',
                            'Q10466', 'Q10598', 'Q10665', 'Q1069', 'Q10743', 'Q10800', 'Q10881', 'Q11031', 'Q11433',
                            'Q11460', 'Q1147', 'Q11772',
                            'Q12090', 'Q12115', 'Q12140', 'Q12172', 'Q1236', 'Q12422', 'Q12558', 'Q12612', 'Q12672',
                            'Q12826', 'Q12955', 'Q1299',
                            'Q13004', 'Q13011', 'Q13029', 'Q13162', 'Q13394', 'Q13436', 'Q13698', 'Q13728', 'Q13880',
                            'Q13965', 'Q14148', 'Q14792',
                            'Q14846', 'Q1488', 'Q15102', 'Q15356', 'Q15369', 'Q15391', 'Q15464', 'Q15498', 'Q15849',
                            'Q15924', 'Q15925', 'Q16037',
                            'Q16228', 'Q16233', 'Q16647', 'Q16713', 'Q16774', 'Q17039', 'Q17364', 'Q17517', 'Q17664',
                            'Q1800', 'Q18088', 'Q18248',
                            'Q18259', 'Q18316', 'Q18394', 'Q18399', 'Q1848', 'Q18684', 'Q18732', 'Q18748', 'Q19396',
                            'Q19419', 'Q19513', 'Q19523',
                            'Q19528', 'Q19600', 'Q19791', 'Q19806', 'Q1991', 'Q1993', 'Q20155', 'Q20178', 'Q20231',
                            'Q2037', 'Q20491', 'Q20560',
                            'Q20586', 'Q20842', 'Q20975', 'Q20980', 'Q21004', 'Q2103', 'Q21054', 'Q2112', 'Q2119',
                            'Q2124', 'Q21744', 'Q21761',
                            'Q21774', 'Q21861', 'Q21974', 'Q21987', 'Q22114', 'Q22152', 'Q22342', 'Q22662', 'Q22876',
                            'Q22943', 'Q23100', 'Q23137',
                            'Q23294', 'Q23308', 'Q23502', 'Q23549', 'Q23694', 'Q23760', 'Q23833', 'Q2416', 'Q2433',
                            'Q2443', 'Q2445', 'Q2780',
                            'Q2784',
                            'Q2922', 'Q2940', 'Q2981', 'Q2998', 'Q3223', 'Q3383', 'Q3446', 'Q3483', 'Q3534', 'Q3604',
                            'Q3606', 'Q363', 'Q3680',
                            'Q3701', 'Q3750', 'Q3855', 'Q3938', 'Q4102', 'Q4308', 'Q4545', 'Q4552', 'Q4713', 'Q47',
                            'Q4895', 'Q4954', 'Q5049',
                            'Q5114', 'Q5278', 'Q5294', 'Q561', 'Q5641', 'Q5678', 'Q5983', 'Q6074', 'Q6082', 'Q6138',
                            'Q6235', 'Q6316', 'Q6355',
                            'Q6363', 'Q6396', 'Q6423', 'Q6490', 'Q6631', 'Q6640', 'Q678', 'Q7034', 'Q7199', 'Q7204',
                            'Q7256', 'Q7307', 'Q7394',
                            'Q7425', 'Q7573', 'Q7757', 'Q7963', 'Q8076', 'Q8191', 'Q8274', 'Q8330', 'Q8345', 'Q8416',
                            'Q8525', 'Q883', 'Q8889',
                            'Q89', 'Q9211', 'Q9687', 'Q9816', 'Q9882', 'Q9977', 'W103', 'W105', 'W106', 'W107', 'W109',
                            'W10', 'W110', 'W111',
                            'W113', 'W116', 'W117', 'W118', 'W119', 'W121', 'W122', 'W123', 'W124', 'W129', 'W130',
                            'W132', 'W135', 'W138',
                            'W139', 'W13', 'W140', 'W142', 'W143', 'W144', 'W145', 'W147', 'W148', 'W149', 'W150',
                            'W151', 'W152', 'W156',
                            'W157', 'W159', 'W15', 'W163', 'W165', 'W16', 'W170', 'W171', 'W177', 'W180', 'W182',
                            'W183', 'W185', 'W187',
                            'W18', 'W190', 'W192', 'W195', 'W196', 'W197', 'W201', 'W202', 'W204', 'W207', 'W208',
                            'W210', 'W211', 'W212',
                            'W216', 'W217', 'W219', 'W21', 'W222', 'W223', 'W224', 'W227', 'W228', 'W229', 'W22',
                            'W230', 'W234', 'W235',
                            'W236', 'W237', 'W23', 'W243', 'W244', 'W245', 'W247', 'W249', 'W24', 'W250', 'W251',
                            'W252', 'W253', 'W254',
                            'W256', 'W259', 'W25', 'W261', 'W262', 'W263', 'W265', 'W266', 'W267', 'W269', 'W270',
                            'W271', 'W272', 'W276',
                            'W277', 'W280', 'W282', 'W283', 'W285', 'W287', 'W288', 'W28', 'W290', 'W291', 'W294',
                            'W295', 'W296', 'W297',
                            'W299', 'W300', 'W301', 'W302', 'W303', 'W304', 'W305', 'W306', 'W307', 'W308', 'W30',
                            'W313', 'W317', 'W318',
                            'W319', 'W320', 'W322', 'W324', 'W325', 'W328', 'W329', 'W32', 'W331', 'W334', 'W335',
                            'W336', 'W337', 'W338',
                            'W340', 'W342', 'W343', 'W344', 'W345', 'W348', 'W352', 'W353', 'W358', 'W359', 'W35',
                            'W38', 'W39', 'W3', 'W40',
                            'W43', 'W44', 'W4', 'W52', 'W53', 'W56', 'W59', 'W60', 'W61', 'W63', 'W64', 'W65', 'W66',
                            'W67', 'W68', 'W69', 'W70',
                            'W72', 'W75', 'W76', 'W78', 'W7', 'W81', 'W82', 'W84', 'W85', 'W87', 'W89', 'W90', 'W92',
                            'W94', 'W95', 'W96', 'W97',
                            'W98', 'W99', 'W9', 'Y10907', 'Y10908', 'Y11041', 'Y1107', 'Y11106', 'Y11116', 'Y11430',
                            'Y13802', 'Y14055',
                            'Y14324', 'Y14689', 'Y15626', 'Y15903', 'Y15920', 'Y16233', 'Y16249', 'Y16315', 'Y1665',
                            'Y16856', 'Y1695', 'Y173',
                            'Y18777', 'Y20444', 'Y20608', 'Y20781', 'Y22287', 'Y22329', 'Y2647', 'Y3207', 'Y3523',
                            'Y3668', 'Y3814', 'Y3836',
                            'Y3917', 'Y4396', 'Y4624', 'Y4640', 'Y4720', 'Y4805', 'Y5087', 'Y509', 'Y5212', 'Y5256',
                            'Y5257', 'Y5354', 'Y5583',
                            'Y5735', 'Y5899', 'Y6057', 'Y6131', 'Y6269', 'Y62', 'Y6574', 'Y6695', 'Y6716', 'Y6752',
                            'Y7134', 'Y7490', 'Y7503',
                            'Y7642', 'Y7690', 'Y7692', 'Y7942', 'Y7943', 'Y7967']
    M4_FORECASTING_BENCH_SMALL = [
        'D1002',
        'D1019',
        'D1032',
        'D1101',
        'D1104',
        'D1124',
        'D1162',
        'D1170',
        'D1204',
        'D1219',
        'M10641',
        'M1080',
        'M11230',
        'M11654',
        'M11779',
        'M11806',
        'M12209',
        'M12241',
        'M12422',
        'M12452',
        'Q10070',
        'Q10262',
        'Q10292',
        'Q10466',
        'Q10598',
        'Q10665',
        'Q1069',
        'Q10743',
        'Q10800',
        'Q10881',
        'W103',
        'W105',
        'W106',
        'W107',
        'W109',
        'W10',
        'W110',
        'W111',
        'W113',
        'W116',
        'Y10907',
        'Y10908',
        'Y11041',
        'Y1107',
        'Y11106',
        'Y11116',
        'Y11430',
        'Y13802',
        'Y14055',
        'Y14324']
    M4_FORECASTING_BENCH_SMALL_DAILY = ['D1002', 'D1019', 'D1032',
                                        'D1101', 'D1104', 'D1124', 'D1162', 'D1170', 'D1204', 'D1219']
    M4_FORECASTING_BENCH_SMALL_MONTHLY = [
        'M10641',
        'M1080',
        'M11230',
        'M11654',
        'M11779',
        'M11806',
        'M12209',
        'M12241',
        'M12422',
        'M12452']
    M4_FORECASTING_BENCH_SMALL_QUARTERLY = ['Q10070', 'Q10262', 'Q10292',
                                            'Q10466', 'Q10598', 'Q10665', 'Q1069', 'Q10743', 'Q10800', 'Q10881']
    M4_FORECASTING_BENCH_SMALL_WEEKLY = [
        'W103', 'W105', 'W106', 'W107', 'W109', 'W10', 'W110', 'W111', 'W113', 'W116']
    M4_FORECASTING_BENCH_SMALL_YEARLY = [
        'Y10907',
        'Y10908',
        'Y11041',
        'Y1107',
        'Y11106',
        'Y11116',
        'Y11430',
        'Y13802',
        'Y14055',
        'Y14324']
    M4_FORECASTING_LENGTH = {'D': 14, 'W': 13, 'M': 18, 'Q': 8, 'Y': 6}
    M4_SEASONALITY = {'D': 1, 'W': 1, 'M': 12, 'Q': 4, 'Y': 1}
    M4_PREFIX = {'D': 'Daily', 'W': 'Weekly',
                 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'}
    MONASH_FORECASTING_BENCH = [
        'australian_electricity_demand',
        'bitcoin',
        'car_parts',
        'cif_2016',
        'covid_deaths',
        'dominick',
        'electricity_hourly',
        'electricity_weekly',
        'fred_md',
        'hospital',
        'kaggle_web_traffic',
        'kdd_cup',
        'm1_monthly',
        'm1_quarterly',
        'm1_yearly',
        'm3_monthly',
        'm3_quarterly',
        'm3_yearly',
        'm4_daily',
        'm4_hourly',
        'm4_monthly',
        'm4_quarterly',
        'm4_weekly',
        'm4_yearly',
        'nn5_daily',
        'nn5_weekly',
        'pedestrian_counts',
        'rideshare',
        'saugeen_river_flow',
        'solar_10_minutes',
        'solar_weekly',
        'sunspot',
        'temperature_rain',
        'tourism_monthly',
        'tourism_quarterly',
        'tourism_yearly',
        'traffic_hourly',
        'traffic_weekly',
        'us_births',
        'vehicle_trips',
        'weather']
    MONASH_FORECASTING_LENGTH = {
        'australian_electricity_demand': 336,
        'bitcoin': 30,
        'car_parts': 12,
        'cif_2016': 12,
        'covid_deaths': 30,
        'dominick': 8,
        'electricity_hourly': 168,
        'electricity_weekly': 8,
        'fred_md': 12,
        'hospital': 12,
        'kaggle_web_traffic': 8,
        'kdd_cup': 168,
        'm1_monthly': 18,
        'm1_quarterly': 8,
        'm1_yearly': 6,
        'm3_monthly': 18,
        'm3_quarterly': 8,
        'm3_yearly': 6,
        'm4_daily': 14,
        'm4_hourly': 48,
        'm4_monthly': 18,
        'm4_quarterly': 8,
        'm4_weekly': 13,
        'm4_yearly': 6,
        'nn5_daily': 56,
        'nn5_weekly': 8,
        'pedestrian_counts': 24,
        'rideshare': 168,
        'saugeen_river_flow': 30,
        'solar_10_minutes': 1008,
        'solar_weekly': 5,
        'sunspot': 30,
        'temperature_rain': 30,
        'tourism_monthly': 24,
        'tourism_quarterly': 8,
        'tourism_yearly': 4,
        'traffic_hourly': 168,
        'traffic_weekly': 8,
        'us_births': 30,
        'vehicle_trips': 30,
        'weather': 30}
    UNI_CLF_BENCH = [
        "ACSF1",
        # "Adiac",  # TODO: fix Adiac dataset infinite composition loop
        "ArrowHead",
        "Beef",
        "BeetleFly",
        "BirdChicken",
        "BME",
        "Car",
        "CBF",
        "Chinatown",
        "ChlorineConcentration",
        "CinCECGTorso",
        "Coffee",
        "Computers",
        "CricketX",
        "CricketY",
        "CricketZ",
        "Crop",
        "DiatomSizeReduction",
        "DistalPhalanxOutlineCorrect",
        "DistalPhalanxOutlineAgeGroup",
        "DistalPhalanxTW",
        "Earthquakes",
        "ECG200",
        "ECG5000",
        "ECGFiveDays",
        "ElectricDevices",
        "EOGHorizontalSignal",
        "EOGVerticalSignal",
        "EthanolLevel",
        "FaceAll",
        "FaceFour",
        "FacesUCR",
        "FiftyWords",
        "Fish",
        "FordA",
        "FordB",
        "FreezerRegularTrain",
        "FreezerSmallTrain",
        "Fungi",
        "GunPoint",
        "GunPointAgeSpan",
        "GunPointMaleVersusFemale",
        "GunPointOldVersusYoung",
        "Ham",
        "HandOutlines",
        "Haptics",
        "Herring",
        "HouseTwenty",
        "InlineSkate",
        "InsectEPGRegularTrain",
        "InsectEPGSmallTrain",
        "InsectWingbeatSound",
        "ItalyPowerDemand",
        "LargeKitchenAppliances",
        "Lightning2",
        "Lightning7",
        "Mallat",
        "Meat",
        "MedicalImages",
        "MiddlePhalanxOutlineCorrect",
        "MiddlePhalanxOutlineAgeGroup",
        "MiddlePhalanxTW",
        "MixedShapesRegularTrain",
        "MixedShapesSmallTrain",
        "MoteStrain",
        "NonInvasiveFetalECGThorax1",
        "NonInvasiveFetalECGThorax2",
        "OliveOil",
        "OSULeaf",
        "PhalangesOutlinesCorrect",
        "Phoneme",
        "PigAirwayPressure",
        "PigArtPressure",
        "PigCVP",
        "Plane",
        "PowerCons",
        "ProximalPhalanxOutlineCorrect",
        "ProximalPhalanxOutlineAgeGroup",
        "ProximalPhalanxTW",
        "RefrigerationDevices",
        "Rock",
        "ScreenType",
        "SemgHandGenderCh2",
        "SemgHandMovementCh2",
        "SemgHandSubjectCh2",
        "ShapeletSim",
        "ShapesAll",
        "SmallKitchenAppliances",
        "SmoothSubspace",
        "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2",
        "StarlightCurves",
        "Strawberry",
        "SwedishLeaf",
        "Symbols",
        "SyntheticControl",
        "ToeSegmentation1",
        "ToeSegmentation2",
        "Trace",
        "TwoLeadECG",
        "TwoPatterns",
        "UMD",
        "UWaveGestureLibraryAll",
        "UWaveGestureLibraryX",
        "UWaveGestureLibraryY",
        "UWaveGestureLibraryZ",
        "Wafer",
        "Wine",
        "WordSynonyms",
        "Worms",
        "WormsTwoClass",
        "Yoga",
    ]
    MULTI_CLF_BENCH = [
        "ArticularyWordRecognition",
        "AtrialFibrillation",
        "BasicMotions",
        "Cricket",
        "DuckDuckGeese",
        "EigenWorms",
        "Epilepsy",
        "EthanolConcentration",
        "ERing",
        "FaceDetection",
        "FingerMovements",
        "HandMovementDirection",
        "Handwriting",
        "Heartbeat",
        "Libras",
        "LSST",
        "MotorImagery",
        "NATOPS",
        "PenDigits",
        "PEMS-SF",
        "PhonemeSpectra",
        "RacketSports",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "StandWalkJump",
        "UWaveGestureLibrary",
    ]


class UnitTestConstant(Enum):
    VALID_LINEAR_CLF_PIPELINE = {
        'frequency_domain_clf': ['industrial_freq_clf'],
        'manifold_clf': ['industrial_manifold_clf'],
        'stat_clf': ['industrial_stat_clf'],
        'eigen_statistical': ['eigen_basis', 'quantile_extractor', 'logit'],
        'channel_filtration_statistical': ['channel_filtration', 'quantile_extractor', 'logit'],
        'fourier_statistical': ['fourier_basis', 'quantile_extractor', 'logit'],
        'wavelet_statistical': ['wavelet_basis', 'quantile_extractor', 'logit'],
        'recurrence_clf': ['recurrence_extractor', 'logit'],
        'riemann_clf': ['riemann_extractor', 'logit'],
        'topological_clf': ['topological_extractor', 'logit'],
        'statistical_clf': ['quantile_extractor', 'logit'],
        # 'statistical_lgbm': ['quantile_extractor', 'lgbm'],
        'composite_clf': {
            0: ['quantile_extractor'], 1: ['riemann_extractor'],
            2: ['fourier_basis', 'quantile_extractor'], 'head': 'mlp'
        },
    }
    VALID_LINEAR_REG_PIPELINE = {
        'stat_reg': ['industrial_stat_reg'],
        'freq_reg': ['industrial_freq_reg'],
        'manifold_reg': ['industrial_manifold_reg'],
        'resnet_reg': ['resnet_model'],
        'inception_reg': ['inception_model'],
        'eigen_statistical_reg': ['eigen_basis', 'quantile_extractor', 'treg'],
        'channel_filtration_statistical_reg': ['channel_filtration', 'quantile_extractor', 'treg'],
        'fourier_statistical_reg': ['fourier_basis', 'quantile_extractor', 'treg'],
        'wavelet_statistical_reg': ['wavelet_basis', 'quantile_extractor', 'treg'],
        'recurrence_reg': ['recurrence_extractor', 'treg'],
        'topological_reg': ['topological_extractor', 'treg'],
        'statistical_reg': ['quantile_extractor', 'treg'],
        # 'statistical_lgbmreg': ['quantile_extractor', 'lgbmreg'],
        'composite_reg': {
            0: ['quantile_extractor'], 1: ['topological_extractor'],
            2: ['fourier_basis', 'quantile_extractor'], 'head': 'treg'
        },
    }
    VALID_LINEAR_TSF_PIPELINE = {
        'stl_arima': ['stl_arima'],
        'topological_lgbm': ['topological_extractor', 'lgbmreg'],
        'ar': ['ar'],
        'eigen_autoregression': ['eigen_forecaster'],
        # 'eigen_autoregression': ['eigen_basis', 'ar'],
        'smoothed_ar': ['smoothing', 'ar'],
        'gaussian_ar': ['gaussian_filter', 'ar'],
        'glm': ['glm'],
        'nbeats': ['nbeats_model'],
        'tcn': ['tcn_model'],
    }
    VALID_LINEAR_DETECTION_PIPELINE = {
        # 'sst': ['sst'],
        # 'unscented_kalman_filter': ['unscented_kalman_filter'],
        'stat_detector': ['stat_detector'],
        'iforest_detector': ['iforest_detector'],
        'conv_ae_detector': ['conv_ae_detector'],
        'lstm_ae_detector': ['lstm_ae_detector'],
        'arima_detector': ['arima_detector'],
    }


STAT_METHODS = FeatureConstant.STAT_METHODS.value
STAT_METHODS_GLOBAL = FeatureConstant.STAT_METHODS_GLOBAL.value
STAT_METHODS_TORCH = FeatureConstant.STAT_METHODS_TORCH.value
STAT_METHODS_GLOBAL_TORCH = FeatureConstant.STAT_METHODS_GLOBAL_TORCH.value
BAGGING_METHOD = FeatureConstant.BAGGING_METHOD.value
PERSISTENCE_DIAGRAM_FEATURES = FeatureConstant.PERSISTENCE_DIAGRAM_FEATURES.value
PERSISTENCE_DIAGRAM_EXTRACTOR = FeatureConstant.PERSISTENCE_DIAGRAM_EXTRACTOR.value
DISCRETE_WAVELETS = FeatureConstant.DISCRETE_WAVELETS.value
CONTINUOUS_WAVELETS = FeatureConstant.CONTINUOUS_WAVELETS.value
WAVELET_SCALES = FeatureConstant.WAVELET_SCALES.value
SINGULAR_VALUE_MEDIAN_THR = FeatureConstant.SINGULAR_VALUE_MEDIAN_THR.value
SINGULAR_VALUE_BETA_THR = FeatureConstant.SINGULAR_VALUE_BETA_THR
DISTANCE_METRICS = FeatureConstant.METRICS_DICT.value
SPECTRUM_ESTIMATORS = FeatureConstant.SPECTRUM_ESTIMATORS.value
SPECTRUM_ESTIMATORS_TORCH = FeatureConstant.SPECTRUM_ESTIMATORS_TORCH.value
DEFAULT_ESTIMATOR_PARAMETERS = FeatureConstant.DEFAULT_ESTIMATOR_PARAMETERS.value

KERNEL_ALGO = KernelsConstant.KERNEL_ALGO.value
KERNEL_BASELINE_FEATURE_GENERATORS_TORCH = KernelsConstant.KERNEL_BASELINE_FEATURE_GENERATORS_TORCH.value
KERNEL_BASELINE_FEATURE_GENERATORS = KernelsConstant.KERNEL_BASELINE_FEATURE_GENERATORS.value
KERNEL_BASELINE_NODE_LIST = KernelsConstant.KERNEL_BASELINE_NODE_LIST.value
KERNEL_DISTANCE_METRIC = KernelsConstant.KERNEL_DISTANCE_METRIC.value

SOLVER_MODELS = SolverConstant.SOLVER_MODELS.value
DEFAULT_SVD_SOLVER = SOLVER_MODELS['np_svd_solver']
DEFAULT_SVD_SOLVER_TORCH = SOLVER_MODELS['torch_svd_solver']
DEFAULT_QR_SOLVER = SOLVER_MODELS['np_qr_solver']
DEFAULT_QR_SOLVER_TORCH = SOLVER_MODELS['torch_qr_solver']
DASK_SVD_SOLVER = SOLVER_MODELS['dask_svd_solver']

AVAILABLE_ANOMALY_DETECTION_OPERATIONS = FedotOperationConstant.AVAILABLE_ANOMALY_DETECTION_OPERATIONS.value
AVAILABLE_REG_OPERATIONS = FedotOperationConstant.AVAILABLE_REG_OPERATIONS.value
AVAILABLE_CLS_OPERATIONS = FedotOperationConstant.AVAILABLE_CLS_OPERATIONS.value
EXCLUDED_OPERATION = FedotOperationConstant.EXCLUDED_OPERATION.value
FEDOT_HEAD_ENSEMBLE = FedotOperationConstant.FEDOT_HEAD_ENSEMBLE.value
FEDOT_TASK = FedotOperationConstant.FEDOT_TASK.value
FEDOT_ATOMIZE_OPERATION = FedotOperationConstant.FEDOT_ATOMIZE_OPERATION.value
FEDOT_GET_METRICS = FedotOperationConstant.FEDOT_GET_METRICS.value
FEDOT_TUNING_METRICS = FedotOperationConstant.FEDOT_TUNING_METRICS.value
FEDOT_ASSUMPTIONS = FedotOperationConstant.FEDOT_ASSUMPTIONS.value
FEDOT_API_PARAMS = FedotOperationConstant.FEDOT_API_PARAMS.value
FEDOT_ENSEMBLE_ASSUMPTIONS = FedotOperationConstant.FEDOT_ENSEMBLE_ASSUMPTIONS.value
FEDOT_TUNER_STRATEGY = FedotOperationConstant.FEDOT_TUNER_STRATEGY.value
FEDOT_INDUSTRIAL_STRATEGY = FedotOperationConstant.FEDOT_INDUSTRIAL_STRATEGY.value
FEDOT_TS_FORECASTING_ASSUMPTIONS = FedotOperationConstant.FEDOT_TS_FORECASTING_ASSUMPTIONS.value
FEDOT_DATA_TYPE = FedotOperationConstant.FEDOT_DATA_TYPE.value
FEDOT_MUTATION_STRATEGY = FedotOperationConstant.FEDOT_MUTATION_STRATEGY.value
EXPLAINABLE_MODELS = FedotOperationConstant.EXPLAINABLE_MODELS.value
SKLEARN_CLF_IMP = FedotOperationConstant.SKLEARN_CLF_MODELS.value
SKLEARN_REG_IMP = FedotOperationConstant.SKLEARN_REG_MODELS.value

CPU_NUMBERS = ComputationalConstant.CPU_NUMBERS.value
BATCH_SIZE_FOR_FEDOT_WORKER = ComputationalConstant.BATCH_SIZE_FOR_FEDOT_WORKER.value
FEDOT_WORKER_NUM = ComputationalConstant.FEDOT_WORKER_NUM.value
FEDOT_WORKER_TIMEOUT_PARTITION = ComputationalConstant.FEDOT_WORKER_TIMEOUT_PARTITION.value
PATIENCE_FOR_EARLY_STOP = ComputationalConstant.PATIENCE_FOR_EARLY_STOP.value

MULTI_ARRAY = DataTypeConstant.MULTI_ARRAY.value
MATRIX = DataTypeConstant.MATRIX.value
TRAJECTORY_MATRIX = DataTypeConstant.TRAJECTORY_MATRIX.value

IND_MODEL_OPERATION_PATH = PathConstant.IND_MODEL_OPERATION_PATH.value
IND_DATA_OPERATION_PATH = PathConstant.IND_DATA_OPERATION_PATH.value
DEFAULT_DATA_OPERATION_PATH = PathConstant.DEFAULT_DATA_OPERATION_PATH.value
DEFAULT_MODEL_OPERATION_PATH = PathConstant.DEFAULT_MODEL_OPERATION_PATH.value

ENERGY_THR = ModelCompressionConstant.ENERGY_THR.value
DECOMPOSE_MODE = ModelCompressionConstant.DECOMPOSE_MODE.value
FORWARD_MODE = ModelCompressionConstant.FORWARD_MODE.value
HOER_LOSS = ModelCompressionConstant.HOER_LOSS.value
ORTOGONAL_LOSS = ModelCompressionConstant.ORTOGONAL_LOSS.value
MODELS_FROM_LENGTH = ModelCompressionConstant.MODELS_FROM_LENGTH.value

CROSS_ENTROPY = TorchLossesConstant.CROSS_ENTROPY.value
MULTI_CLASS_CROSS_ENTROPY = TorchLossesConstant.MULTI_CLASS_CROSS_ENTROPY.value
MSE = TorchLossesConstant.MSE.value
RMSE = TorchLossesConstant.RMSE.value
SMAPE = TorchLossesConstant.SMAPE.value
TWEEDIE_LOSS = TorchLossesConstant.TWEEDIE_LOSS.value
FOCAL_LOSS = TorchLossesConstant.FOCAL_LOSS.value
CENTER_PLUS_LOSS = TorchLossesConstant.CENTER_PLUS_LOSS.value
CENTER_LOSS = TorchLossesConstant.CENTER_LOSS.value
MASK_LOSS = TorchLossesConstant.MASK_LOSS.value
LOG_COSH_LOSS = TorchLossesConstant.LOG_COSH_LOSS.value
HUBER_LOSS = TorchLossesConstant.HUBER_LOSS.value
EXPONENTIAL_WEIGHTED_LOSS = TorchLossesConstant.EXPONENTIAL_WEIGHTED_LOSS.value

MULTI_REG_BENCH = BenchmarkDatasets.MULTI_REG_BENCH.value
UNI_CLF_BENCH = BenchmarkDatasets.UNI_CLF_BENCH.value
MULTI_CLF_BENCH = BenchmarkDatasets.MULTI_CLF_BENCH.value
M4_FORECASTING_BENCH = BenchmarkDatasets.M4_FORECASTING_BENCH.value
M4_FORECASTING_BENCH_SMALL = BenchmarkDatasets.M4_FORECASTING_BENCH_SMALL.value
M4_FORECASTING_BENCH_SMALL_DAILY = BenchmarkDatasets.M4_FORECASTING_BENCH_SMALL_DAILY.value
M4_FORECASTING_BENCH_SMALL_MONTHLY = BenchmarkDatasets.M4_FORECASTING_BENCH_SMALL_MONTHLY.value
M4_FORECASTING_BENCH_SMALL_QUARTERLY = BenchmarkDatasets.M4_FORECASTING_BENCH_SMALL_QUARTERLY.value
M4_FORECASTING_BENCH_SMALL_WEEKLY = BenchmarkDatasets.M4_FORECASTING_BENCH_SMALL_WEEKLY.value
M4_FORECASTING_BENCH_SMALL_YEARLY = BenchmarkDatasets.M4_FORECASTING_BENCH_SMALL_YEARLY.value
M4_FORECASTING_LENGTH = BenchmarkDatasets.M4_FORECASTING_LENGTH.value
M4_SEASONALITY = BenchmarkDatasets.M4_SEASONALITY.value
MONASH_FORECASTING_BENCH = BenchmarkDatasets.MONASH_FORECASTING_BENCH.value
MONASH_FORECASTING_LENGTH = BenchmarkDatasets.MONASH_FORECASTING_LENGTH.value
M4_PREFIX = BenchmarkDatasets.M4_PREFIX.value

VALID_LINEAR_CLF_PIPELINE = UnitTestConstant.VALID_LINEAR_CLF_PIPELINE.value
VALID_LINEAR_REG_PIPELINE = UnitTestConstant.VALID_LINEAR_REG_PIPELINE.value
VALID_LINEAR_TSF_PIPELINE = UnitTestConstant.VALID_LINEAR_TSF_PIPELINE.value
VALID_LINEAR_DETECTION_PIPELINE = UnitTestConstant.VALID_LINEAR_DETECTION_PIPELINE.value


def fedot_init_assumptions(problem):
    return FEDOT_ASSUMPTIONS[problem]


def fedot_task(task, task_params: dict = None):
    fedot_task = FEDOT_TASK[task]
    if task_params is not None:
        fedot_task.task_params.forecast_length = task_params
        return fedot_task
    else:
        return fedot_task
