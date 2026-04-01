import logging
import os

import matplotlib
from matplotlib import pyplot as plt

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader


class AbstractBenchmark(object):
    """Abstract class for benchmarks.

    This class defines the interface that all benchmarks must implement.
    """

    def __init__(self, output_dir, **kwargs):
        """Initialize the benchmark.

        Args:
            name: The name of the benchmark.
            description: A short description of the benchmark.
            **kwargs: Additional arguments that may be required by the
                benchmark.
        """
        self.result_dir = output_dir
        self.kwargs = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
        self._create_output_dir()

    @property
    def _config(self):
        raise NotImplementedError()

    def _create_output_dir(self):
        os.makedirs(self.result_dir, exist_ok=True)

    def _create_report(self, results):
        """Create a report from the results of the benchmark.

        Args:
            results: The results of the benchmark.

        Returns:
            A string containing the report.
        """
        raise NotImplementedError()

    def run(self):
        """Run the benchmark and return the results.

        Returns:
            A dictionary containing the results of the benchmark.
        """
        raise NotImplementedError()

    def evaluate_loop(self, dataset, experiment_setup: dict = None):
        matplotlib.use('TkAgg')
        train_data, test_data = DataLoader(dataset_name=dataset).load_data()
        experiment_setup['compute_config']['output_folder'] += f'/{dataset}'
        experiment_setup['compute_config']['history_dir'] = f'./composition_results/{dataset}'
        model = FedotIndustrial(**experiment_setup)
        model.fit(train_data)
        prediction = model.predict(test_data)
        model.save_best_model()
        model.save_optimization_history()
        # model.plot_operation_distribution(mode='each')
        # model.plot_fitness_by_generation()
        plt.close('all')
        model.shutdown()
        model.return_report()
        return prediction.squeeze(), model.predict_data.target

    def finetune_loop(self, dataset, experiment_setup, composed_model_path):
        train_data, test_data = DataLoader(dataset_name=dataset).load_data()
        model = FedotIndustrial(**experiment_setup)
        model.load(path=composed_model_path)
        model.finetune(train_data)
        prediction = model.predict(test_data)
        return prediction, model

    def collect_results(self, output_dir):
        """Collect the results of the benchmark.

        Args:
            output_dir: The directory where the benchmark wrote its results.

        Returns:
            A dictionary containing the results of the benchmark.
        """
        raise NotImplementedError()
