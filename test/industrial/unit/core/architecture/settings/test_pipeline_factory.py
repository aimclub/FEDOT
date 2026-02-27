from fedot_ind.core.architecture.settings.pipeline_factory import BasisTransformations, FeatureGenerator, MlModel, KernelFeatureGenerator


def test_basis_transformations():
    assert BasisTransformations.datadriven is not None
    assert BasisTransformations.wavelet is not None
    assert BasisTransformations.Fourier is not None


def test_feature_generator():
    assert FeatureGenerator.quantile is not None
    assert FeatureGenerator.topological is not None
    assert FeatureGenerator.recurrence is not None


def test_ml_model():
    assert MlModel.functional_pca is not None
    assert MlModel.kalman_filter is not None
    assert MlModel.sst is not None


def test_kernel_feature_generator():
    assert KernelFeatureGenerator.quantile is not None
    assert KernelFeatureGenerator.wavelet is not None
