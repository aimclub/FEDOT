from copy import deepcopy

import pandas as pd

from fedot.api.main import Fedot
from fedot.core.operations.atomized_model import AtomizedModel
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root


def run_additional_learning_example():
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    test_data_target = test_data['target']
    del test_data['target']

    problem = 'classification'

    auto_model = Fedot(problem=problem, seed=42, preset='light', learning_time=5,
                       composer_params={'initial_chain':
                                        Pipeline(SecondaryNode('logit', nodes_from=[PrimaryNode('scaling')]))})
    auto_model.fit(features=deepcopy(train_data.head(1000)), target='target')
    auto_model.predict_proba(features=deepcopy(test_data))
    print('auto_model', auto_model.get_metrics(target=deepcopy(test_data_target)))

    prev_model = auto_model.current_model
    prev_model.show()

    prev_model.unfit()
    atomized_model = Pipeline(
        SecondaryNode(operation_type=AtomizedModel(prev_model), nodes_from=[PrimaryNode('scaling')]))
    non_atomized_model = deepcopy(prev_model)

    train_data = train_data.head(5000)
    learning_time = 1

    auto_model_from_atomized = Fedot(problem=problem, seed=42, preset='light', learning_time=learning_time,
                                     composer_params={'initial_chain': atomized_model}, verbose_level=2)
    auto_model_from_atomized.fit(features=deepcopy(train_data), target='target')
    auto_model_from_atomized.predict_proba(features=deepcopy(test_data))
    auto_model_from_atomized.current_model.show()
    print('auto_model_from_atomized', auto_model_from_atomized.get_metrics(deepcopy(test_data_target)))

    auto_model_from_chain = Fedot(problem=problem, seed=42, preset='light', learning_time=learning_time,
                                  composer_params={'initial_chain': non_atomized_model}, verbose_level=2)
    auto_model_from_chain.fit(features=deepcopy(train_data), target='target')
    auto_model_from_chain.predict_proba(features=deepcopy(test_data))
    auto_model_from_chain.current_model.show()
    print('auto_model_from_chain', auto_model_from_chain.get_metrics(deepcopy(test_data_target)))


if __name__ == '__main__':
    run_additional_learning_example()
