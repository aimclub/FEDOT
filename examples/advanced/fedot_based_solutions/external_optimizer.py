import logging

from golem.core.optimisers.random.random_mutation_optimizer import RandomMutationSearchOptimizer

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root


def run_with_random_search_composer():
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    composer_params = {'available_operations': ['class_decompose', 'rf', 'linear', 'xgboost', 'dt'],
                       'optimizer': RandomMutationSearchOptimizer}

    automl = Fedot(problem='classification', timeout=1, logging_level=logging.DEBUG,
                   preset='fast_train', **composer_params)

    automl.fit(train_data_path)
    automl.predict(test_data_path)
    print(automl.get_metrics())


if __name__ == '__main__':
    run_with_random_search_composer()
