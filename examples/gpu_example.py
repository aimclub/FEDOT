from datetime import datetime
from typing import Tuple
from os.path import join
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root, default_fedot_data_dir
from sklearn.datasets import make_moons

from matplotlib import pyplot as plt


def get_pipeline():
    svc_node_with_custom_params = PrimaryNode('svc')
    svc_node_with_custom_params.custom_params = dict(kernel='rbf', C=10,
                                                     gamma=1, cache_size=2000,
                                                     probability=True)

    svc_primary_node = PrimaryNode('svc')
    svc_primary_node.custom_params = dict(probability=True)

    knn_primary_node = PrimaryNode('knn')
    logit_secondary_node = SecondaryNode('logit', nodes_from=[svc_primary_node])

    knn_secondary_node = SecondaryNode('knn', nodes_from=[knn_primary_node, logit_secondary_node])

    logit_secondary_node = SecondaryNode('logit')

    rf_node = SecondaryNode('rf', nodes_from=[logit_secondary_node, knn_secondary_node])

    preset_pipeline = Pipeline(rf_node)

    return preset_pipeline


def run_one_model_with_specific_evaluation_mod(train_data, test_data, mode: str = None):
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


def run_pipeline_with_specific_evaluation_mode(train_data: InputData, test_data: InputData,
                                               n_times: list = None, samples: int = None,
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

    preset_pipeline = get_pipeline()

    start = datetime.now()
    baseline_model.fit(features=train_data, target='target', predefined_model=preset_pipeline)
    finish = datetime.now() - start
    print(f'Completed in: {finish}')
    if samples:
        print(f'Finish {samples} params in: {finish}')
    if n_times:
        total_time = float(f'{finish.seconds}.{finish.microseconds}')
        n_times.append(total_time)

    baseline_model.predict(features=test_data)
    print(baseline_model.get_metrics())
    return n_times


def get_scoring_data() -> Tuple[InputData, InputData]:
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    train_data = InputData.from_csv(train_data_path)
    test_data = InputData.from_csv(test_data_path)

    return train_data, test_data


def make_moons_input_data(samples):
    X, y = make_moons(n_samples=samples, shuffle=True, noise=0.1, random_state=137)

    train_data = InputData(idx=list(range(len(X))), features=X, target=y,
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    test_data = InputData(idx=list(range(len(X))), features=X, target=y,
                          task=Task(TaskTypesEnum.classification),
                          data_type=DataTypesEnum.table)

    return train_data, test_data


def draw_plot(cpu_times, gpu_times, n_samples):
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.rcParams['font.size'] = '16'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    ax.plot(n_samples, cpu_times, label='cpu', marker='.')
    ax.plot(n_samples, gpu_times, label='gpu', marker='.')
    ax.set_ylabel('time, s', fontsize=16)
    ax.set_xlabel('n_samples', fontsize=16)
    leg = ax.legend(prop={"size": 16})
    plt.savefig(join(default_fedot_data_dir(), 'gpu_comparison.jpg'))


def run_moons_sample_comparison_example():
    # n_samples = [10000, 100000, 200000, 300000, 400000, 500000]
    # n_samples = [10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    n_samples = [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    n_times_gpu = []
    n_times_cpu = []
    for samples in n_samples:
        train_data, test_data = make_moons_input_data(samples)
        n_times_cpu = run_pipeline_with_specific_evaluation_mode(train_data=train_data, test_data=test_data,
                                                                 samples=samples, n_times=n_times_cpu)
    for samples in n_samples:
        train_data, test_data = make_moons_input_data(samples)
        n_times_gpu = run_pipeline_with_specific_evaluation_mode(train_data=train_data, test_data=test_data,
                                                                 mode='gpu', samples=samples, n_times=n_times_gpu)

    draw_plot(n_times_cpu, n_times_gpu, n_samples)


if __name__ == '__main__':
    train_data, test_data = get_scoring_data()

    run_one_model_with_specific_evaluation_mod(train_data=train_data, test_data=test_data,
                                               mode='gpu')

    run_pipeline_with_specific_evaluation_mode(train_data=train_data, test_data=test_data,
                                               mode='gpu')
