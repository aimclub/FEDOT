import logging

from fedot.industrial.core.architecture.settings.computational import backend_methods as np


class MetaFeaturesDetector:
    """Class for detecting metafeatures of the dataset.

    Args:
        train_data: Training data in tuple format (X, y).
        test_data: Test data in tuple

    Attributes:
        train_data: Training data in tuple format (X, y).
        test_data: Test data in tuple
        dataset_name: Name of the dataset.
        logger (logging.Logger): Logger instance for logging messages.
        base_metafeatures (list): List of base metafeatures.
        extra_metafeatures (list): List of extra metafeatures.

    """

    def __init__(self, train_data, test_data, dataset_name):
        self.train_data = train_data
        self.test_data = test_data
        self.dataset_name = dataset_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f'Initializing MetaFeaturesDetector for {dataset_name}')

        self.base_metafeatures = [
            'test_size', 'train_size', 'length', 'number_of_classes', 'type']
        self.extra_metafeatures = []

    def get_base_metafeatures(self):
        train_features_dict = {}
        ts_df_train, ts_target_train = self.train_data[0], self.train_data[1]
        ts_df_test, ts_target_test = self.test_data[0], self.test_data[1]

        train_size = len(ts_target_train)
        test_size = len(ts_target_test)
        num_classes = len(np.unique(ts_target_train))
        length = ts_df_train.shape[1]
        _type = 'custom'
        train_features_dict['train_size'] = train_size
        train_features_dict['test_size'] = test_size
        train_features_dict['number_of_classes'] = num_classes
        train_features_dict['length'] = length
        train_features_dict['type'] = _type

        return train_features_dict

    def get_extra_metafeatures(self):
        pass

    def run(self):
        self.logger.info(
            f'Running MetaFeaturesDetector for {self.dataset_name}')
        base_metafeatures = self.get_base_metafeatures()
        return {**base_metafeatures}
