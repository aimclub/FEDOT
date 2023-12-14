import logging
from copy import deepcopy

import pandas as pd

from fedot import Fedot
from fedot.core.operations.atomized_model.atomized_model import AtomizedModel
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.utils import fedot_project_root
from fedot.core.utils import set_random_seed


def run_additional_learning_example():
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    test_data_target = test_data['target']
    del test_data['target']

    problem = 'classification'

    auto_model = Fedot(problem=problem, timeout=5, preset='best_quality',
                       initial_assumption=PipelineBuilder().add_node('scaling').add_node('logit').build())

    auto_model.fit(features=deepcopy(train_data.head(1000)), target='target')
    auto_model.predict_proba(features=deepcopy(test_data))
    print('auto_model', auto_model.get_metrics(target=deepcopy(test_data_target)))

    prev_model = auto_model.current_pipeline
    prev_model.show()

    prev_model.unfit()
    atomized_model = Pipeline(
        PipelineNode(operation_type=AtomizedModel(prev_model), nodes_from=[PipelineNode('scaling')]))
    non_atomized_model = deepcopy(prev_model)

    train_data = train_data.head(5000)
    timeout = 1

    auto_model_from_atomized = Fedot(problem=problem, preset='best_quality', timeout=timeout,
                                     logging_level=logging.FATAL,
                                     initial_assumption=atomized_model)
    auto_model_from_atomized.fit(features=deepcopy(train_data), target='target')
    auto_model_from_atomized.predict_proba(features=deepcopy(test_data))
    auto_model_from_atomized.current_pipeline.show()
    print('auto_model_from_atomized', auto_model_from_atomized.get_metrics(deepcopy(test_data_target)))

    auto_model_from_pipeline = Fedot(problem=problem, preset='best_quality', timeout=timeout,
                                     logging_level=logging.FATAL,
                                     initial_assumption=non_atomized_model)
    auto_model_from_pipeline.fit(features=deepcopy(train_data), target='target')
    auto_model_from_pipeline.predict_proba(features=deepcopy(test_data))
    auto_model_from_pipeline.current_pipeline.show()
    print('auto_model_from_pipeline', auto_model_from_pipeline.get_metrics(deepcopy(test_data_target)))


if __name__ == '__main__':
    set_random_seed(42)

    run_additional_learning_example()
