import pandas as pd
from sklearn.model_selection import train_test_split

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.tools.synthetic.ts_generator import TimeSeriesGenerator


class TimeSeriesDatasetsGenerator:
    """
    Generates dummy time series datasets for classification tasks.

    Args:
        num_samples: The number of samples to generate.
        max_ts_len: The maximum length of the time series.
        binary: Whether to generate binary classification datasets or multiclass.
        test_size : The proportion of the dataset to include in the test split.
        multivariate: Whether to generate multivariate time series.

    Example:
        Easy::

            generator = TimeSeriesDatasetsGenerator(num_samples=80,
                                                    task='classification',
                                                    max_ts_len=50,
                                                    binary=True,
                                                    test_size=0.5,
                                                    multivariate=False)
            train_data, test_data = generator.generate_data()

    """

    def __init__(self,
                 task: str = 'classification',
                 num_samples: int = 80,
                 max_ts_len: int = 50,
                 binary: bool = True,
                 test_size: float = 0.5,
                 multivariate: bool = False):
        self.task = task
        self.num_samples = num_samples
        self.max_ts_len = max_ts_len
        self.test_size = test_size
        self.multivariate = multivariate

        if binary:
            self.selected_classes = ['sin', 'random_walk']
        else:
            self.selected_classes = ['sin', 'random_walk', 'auto_regression']

    def generate_data(self):
        """
        Generates the dataset and returns it as a tuple of train and test data.

        Returns:
            Tuple of train and test data, each containing tuples of features and targets.

        """
        if self.multivariate:
            n_classes = len(self.selected_classes)
            features = self.create_features(
                self.num_samples * n_classes,
                self.max_ts_len,
                self.multivariate)

            if self.task == 'classification':
                target = np.random.randint(
                    0, n_classes, self.num_samples * n_classes)
            else:
                target = np.random.randn(self.num_samples * n_classes)
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=self.test_size, random_state=42, shuffle=True)
            return (X_train, y_train), (X_test, y_test)

        ts_frame = pd.DataFrame()
        labels = np.array([])
        for idx, ts_class in enumerate(self.selected_classes):
            for sample in range(self.num_samples):
                if self.task == 'classification':
                    label = idx
                else:
                    label = np.random.randn()
                params = {'ts_type': ts_class,
                          'length': self.max_ts_len}
                ts_gen = TimeSeriesGenerator(params)
                ts = ts_gen.get_ts()
                ts_frame = pd.concat(
                    [ts_frame, pd.DataFrame(ts).T], ignore_index=True)
                labels = np.append(labels, label)
        ts_frame.reset_index(drop=True, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(
            ts_frame, labels, test_size=self.test_size, random_state=42, shuffle=True)
        return (X_train, y_train), (X_test, y_test)

    def create_features(self, n_samples, ts_length, multivariate):
        features = pd.DataFrame(np.random.random((n_samples, ts_length)))
        # TODO: add option to select dimensions
        if multivariate:
            features = np.random.rand(n_samples, 3, ts_length)
        return features
