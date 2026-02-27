import pytest

from fedot_ind.tools.explain.metafeatures import MetaFeaturesDetector
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


@pytest.fixture(scope='session')
def dataset():
    return TimeSeriesDatasetsGenerator().generate_data()


def test_MetaFeaturesDetector(dataset):
    train_data, test_data = dataset
    detector = MetaFeaturesDetector(train_data, test_data, 'test_dataset')
    assert detector is not None
    assert detector.train_data == train_data
    assert detector.test_data == test_data
    assert detector.dataset_name == 'test_dataset'


def test_get_base_metafeatures(dataset):
    train_data, test_data = dataset
    detector = MetaFeaturesDetector(train_data, test_data, 'test_dataset')
    train_features_dict = detector.get_base_metafeatures()
    assert train_features_dict is not None
    assert all(
        [feature in detector.base_metafeatures for feature in train_features_dict.keys()])


def test_get_extra_metafeatures(dataset):
    pass


def test_run(dataset):
    detector = MetaFeaturesDetector(dataset[0], dataset[1], 'test_dataset')
    metafeatures_dict = detector.run()
    expected_features = detector.base_metafeatures + detector.extra_metafeatures
    assert metafeatures_dict is not None
    assert all([f in expected_features for f in metafeatures_dict.keys()])
