import os
import sys
from datetime import datetime

from typing import Tuple

curdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(curdir, '..'))
ROOT = os.path.abspath(os.curdir)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "fedot"))

from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root


def run_one_model_gpu_example(train_data, test_data, mode: str = None):
    """
    Runs the example with one model svc.

    :param train_data: train data for pipeline training
    :param test_data: test data for pipeline training
    :param mode: pass gpu flag to make gpu evaluation
    """

    problem = 'classification'

    if mode == 'gpu':
        baseline_model = Fedot(problem=problem, preset='gpu')
    else:
        baseline_model = Fedot(problem=problem)
    svc_node_with_custom_params = PrimaryNode('svc')
    # the custom params are needed to make probability evaluation available
    # otherwise an error is occurred
    svc_node_with_custom_params.custom_params = dict(kernel='rbf', C=10, gamma=1, cache_size=2000, probability=True)
    preset_pipeline = Pipeline(svc_node_with_custom_params)

    start = datetime.now()
    baseline_model.fit(features=train_data, target='target', predefined_model=preset_pipeline)
    print(f'Completed with custom params in: {datetime.now() - start}')

    baseline_model.predict(features=test_data)
    print(baseline_model.get_metrics())


def run_pipeline_gpu_example(train_data: InputData, test_data: InputData,
                             mode: str = None):
    """
    Runs the example with 3-node pipeline.

    :param train_data: train data for pipeline training
    :param test_data: test data for pipeline training
    :param mode: pass gpu flag to make gpu evaluation
    """
    problem = 'classification'

    if mode == 'gpu':
        baseline_model = Fedot(problem=problem, preset='gpu')
    else:
        baseline_model = Fedot(problem=problem)

    svc_node_with_custom_params = PrimaryNode('svc')
    svc_node_with_custom_params.custom_params = dict(kernel='rbf', C=10, gamma=1, cache_size=2000, probability=True)

    logit_node = PrimaryNode('logit')

    rf_node = SecondaryNode('rf', nodes_from=[svc_node_with_custom_params, logit_node])

    preset_pipeline = Pipeline(rf_node)

    start = datetime.now()
    baseline_model.fit(features=train_data, target='target', predefined_model=preset_pipeline)
    print(f'Completed with custom params in: {datetime.now() - start}')

    baseline_model.predict(features=test_data)
    print(baseline_model.get_metrics())


def get_scoring_data() -> Tuple[InputData, InputData]:
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    train_data = InputData.from_csv(train_data_path)
    test_data = InputData.from_csv(test_data_path)

    return train_data, test_data


if __name__ == '__main__':
    train_data, test_data = get_scoring_data()

    run_one_model_gpu_example(train_data=train_data, test_data=test_data,
                              mode='gpu')
    run_pipeline_gpu_example(train_data=train_data, test_data=test_data,
                             mode='gpu')
