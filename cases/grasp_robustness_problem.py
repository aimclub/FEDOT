import datetime
import os
import random
from core.composer.node import PrimaryNode, SecondaryNode
from core.composer.chain import Chain

from core.composer.metrics import AccuracyScore, F1Metric, PrecisionMetric, RecallMetric, RocAucMetric
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.composer.visualisation import ComposerVisualiser
from core.models.data import InputData, train_test_data_setup
from core.repository.model_types_repository import (
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from core.repository.tasks import Task, TaskTypesEnum
from core.utils import project_root

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

random.seed(1)
np.random.seed(1)


def copy(data: InputData) -> InputData:
    """
    Make copy of passed data.

    :param data: InputData class instance data will be copied.
    :return: InputData
    """
    data_copied = InputData(idx=data.idx, features=data.features, target=data.target,
                            task=data.task, data_type=data.data_type)
    return data_copied


def create_performance_model(dataset: InputData, path_to_save: str,
                             time_limit: int, list_of_nodes: list,
                             percent: float, top_percent: float,
                             percent_step: float, feature_top=None, feature_step: int = 1,
                             path_to_save_figure: str = None) -> pd.DataFrame:
    """
    Visualise function time=f(dataset_size, features_number).

    :param dataset: InputData instance which used to fit given chain.
    :param path_to_save: Path to save csv-file with the return value.
    :param time_limit: Time limit of execution in minutes.
    :param list_of_nodes: List of model_id strings or a single model_id string
        from which the fixed structure chain will be constructed. The last model in the list_of_nodes is a chain root,
        the others are his children. Max chain depth is equal to 2.
    :param percent: The low border of the input data to start tuning. It has to be float in range from 0.0 to 1.0.
        It should be mentioned that this is not initial data to fit because of the prior data splitting into train and
        validation samples.
    :param top_percent: The top border of the input data to finish tuning. It has to be float in range from 0.0 to 1.0
        and more than param percent. It should be mentioned that this is not final data to fit because of the prior
        data splitting into train and validation samples.
    :param percent_step: Step in dataset size. It has to be float in range from 0.0 to 1.0.
    :param feature_top: The top limit of features to be used. The features are chosen consistently in the order
        defined by input data features from left to right.
    :param feature_step: Step in features. Default value is 1.
    :param path_to_save_figure: Path to save visualisation result. The default value is None which means that
        the plot will not be saved in any file but visualisation will be available in the console.
    :return: pd.DataFrame: table with the following columns: 'time', 'num_lines', 'num_features', 'roc_auc',
                                         'accuracy', 'f1', 'precision', 'recall'.
    """
    initial_time = datetime.datetime.now()
    current_time = 0
    arr = []
    features_count = dataset.features.shape[1]
    if feature_top is None:
        feature_top = features_count
    if feature_top > features_count:
        raise ValueError('Invalid value of param feature_top')
    dataset_original = copy(dataset)
    initial_percent = percent
    for i in np.arange(1, feature_top+1, feature_step):
        dataset.features = dataset_original.features[:, :i]
        while current_time <= time_limit and percent <= top_percent:
            # decreasing number of dataset lines
            dat, _ = train_test_data_setup(dataset, split_ratio=percent)

            # split dataset to train and test sets
            dataset_to_compose, dataset_to_validate = train_test_data_setup(dat)
            num_lines = dataset_to_compose.target.shape[0]

            # calculate parameters optimization time
            chain = fixed_chain(list_of_nodes)
            start_time = datetime.datetime.now()
            chain.fit(input_data=dataset_to_compose, verbose=True)
            time = datetime.datetime.now() - start_time
            roc_on_valid_evo_composed = RocAucMetric.get_value(chain, dataset_to_validate)
            acc_on_valid_evo_composed = AccuracyScore.get_value(chain, dataset_to_validate)
            f1_on_valid_evo_composed = F1Metric.get_value(chain, dataset_to_validate)
            precis_on_valid_evo_composed = PrecisionMetric.get_value(chain, dataset_to_validate)
            recall_on_valid_evo_composed = RecallMetric.get_value(chain, dataset_to_validate)
            arr.append([time.total_seconds(), num_lines, i, -roc_on_valid_evo_composed, -acc_on_valid_evo_composed,
                        -f1_on_valid_evo_composed, -precis_on_valid_evo_composed, -recall_on_valid_evo_composed])
            print(f'Dataset size: {percent * 100}%, number of features: {i},'
                  f' required time: {time}, elapsed time: {datetime.datetime.now() - initial_time}')
            print('--------------------')
            percent += percent_step
            current_time = round((datetime.datetime.now() - initial_time).seconds / 60)
        percent = initial_percent
        print('===============================')
    if len(list_of_nodes) == 1:
        print(f'Time is up or process has finished for model: {list_of_nodes[0]}')
    else:
        print(f'Time is up or process has finished for chain: {list_of_nodes}')
    print('===============================')
    results = pd.DataFrame(arr, columns=['time', 'num_lines', 'num_features', 'roc_auc',
                                         'accuracy', 'f1', 'precision', 'recall'])
    results.to_csv(path_to_save, index=False)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(results['num_lines'], results['num_features'], results['time'], c='r')
    ax.set_title(f'Performance model for {list_of_nodes}')
    ax.set_xlabel('num_lines')
    ax.set_ylabel('num_features')
    ax.set_zlabel('time [seconds]')
    ax.grid()
    plt.show()
    if path_to_save_figure:
        fig.savefig(path_to_save_figure)
    return results


def evolution_visualisation(path_to_history_file, path_to_save_image=None,
                            indices: list = None, flag: bool = False) -> None:
    """
    Plot best fitness and average fitness per generation during composition.

    :param path_to_history_file: absolute path to history.csv file location.
    :param path_to_save_image: PC location to save image.
    :param indices: list of dataset lines indices to remove.
    :param flag: if True then remove previous x-tick to prevent x-ticks labels intersection.
    :return: None
    """
    with open(path_to_history_file) as f:
        result = f.read().replace('"', '')
    with open(path_to_history_file, mode='w') as f:
        f.write(result)
    df = pd.read_csv(path_to_history_file)
    num_individuals = df.shape[0]-1
    num_generation = df['generation'].unique().shape[0]-1
    pop_size = int(num_individuals / num_generation)
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[1].set_xlim(left=0, right=num_generation)
    axes[0].set_title('Evolution')
    axes[0].set_ylabel('best_fitness')
    axes[1].set_xlabel('num_of_generation')
    axes[1].set_ylabel('average_fitness')
    x = np.arange(1, num_generation + 1)
    if indices:
        df = df.drop(indices)
    y = df['fitness'].to_numpy()
    arr = np.zeros((2, num_generation))
    count = 0
    for i in range(0, num_individuals, pop_size):
        fitness_max = (-y[i:i + pop_size]).max()
        fitness_average = -y[i:i + pop_size].mean()
        arr[0, count] = fitness_max
        arr[1, count] = fitness_average
        count += 1
    axes[0].plot(x, arr[0, :], color='r')
    axes[1].plot(x, arr[1, :], color='b')
    pos_max = np.argmax(arr[0, :]) + 1
    best = round(arr[0, :].max(), 3)
    res_y = np.hstack((axes[0].get_yticks(), best))
    res_x = np.hstack((axes[1].get_xticks(), pos_max))
    res_y.sort()
    res_x.sort()
    if flag:
        index, = np.where(res_x == pos_max)
        res_x = np.delete(res_x, index-1)
    axes[0].set_yticks(res_y)
    axes[1].set_xticks(res_x)
    axes[0].grid()
    axes[1].grid()
    plt.show()
    if path_to_save_image:
        fig.savefig(path_to_save_image)
    print(f'Visualization was finished')
    return None


def fixed_chain(nodes: list) -> Chain:
    """
    Fixed structure chain creation. Chain can consists of a single model.
    The last model in the list nodes is a chain root, the others are his children.
    Max chain depth is equal to 2.

    :param nodes: List of model_id strings or a single model_id string
        from which the fixed structure chain will be constructed. The last model in the list_of_nodes is a chain root,
        the others are his children. Max chain depth is equal to 2.
    :return: Chain
    """
    if len(nodes) > 1:
        root_of_tree = SecondaryNode(nodes[-1])
        for model_type in nodes[:-1]:
            root_of_tree.nodes_from.append(PrimaryNode(model_type))
        chain_evo_composed = Chain()

        for node in root_of_tree.nodes_from:
            chain_evo_composed.add_node(node)
        chain_evo_composed.add_node(root_of_tree)
    else:
        single_node = PrimaryNode(nodes[0])
        chain_evo_composed = Chain(single_node)
    return chain_evo_composed


def run_grasp_robustness_problem(dataset_path, list_of_nodes: list,
                                 part_of_dataset_to_compose=0.05, full_dataset=False,
                                 max_lead_time: datetime.timedelta = datetime.timedelta(minutes=180),
                                 is_visualise=False, fit_fixed_chain=False) -> Chain:
    """
    Run process of composing and the following chain fit to binary classify manipulator grasp stability.
    Composition is based on genetic programming methods. Generally, the algorithm consists of two steps.
    The first one is to obtain optimal chain during composition which based on a small part of the original
    dataset size due to its time-consuming. The last one is to fit composed chain at 80% percent of dataset size
    to get the best results for 5-th metrics.

    :param dataset_path: Original dataset csv-file location.
    :param list_of_nodes: List of model_id strings or a single model_id string
        from which the fixed structure chain will be constructed. The last model in the list_of_nodes is a chain root,
        the others are his children. Max chain depth is equal to 2.
    :param part_of_dataset_to_compose: Part of original dataset to use in chain composing.
        Default value is equal to 0.05 to accelerate chain composing which is the most time-consuming task.
        It's necessary thing because original dataset size is too big.
        The parameter value upper limit is set to 0.8 because of the validation set size is automatically defined as 20%
        of the original dataset. In oder to use the upper limit value it's need to set param full_dataset to True.
    :param full_dataset: If true then the original dataset is fully used. By default, 80% to compose and fit
        and the last 20% percent to validate.
    :param max_lead_time: Time limit of program execution. The real time of execution can differ from this value due to
        local features.
    :param is_visualise: If true then visualize obtained chain.
    :param fit_fixed_chain: If true then the process consists of only tuning without composition.
    :return: Chain
    """

    if part_of_dataset_to_compose >= 0.8:
        raise ValueError("The argument part_of_dataset_to_compose has to be less than 0.8")
    dataset_train_fit = None
    task = Task(TaskTypesEnum.classification)
    dataset = InputData.from_csv(dataset_path, headers=['measurement_number'], task=task, target_header='robustness')

    # this is a sensible grasp threshold for stability
    good_grasp_threshold = 100

    # divide the grasp quality on stable or unstable grasps
    dataset.target = np.array([int(i > good_grasp_threshold) for i in dataset.target])

    # split dataset to train and test sets
    if full_dataset:
        dataset_to_compose, dataset_to_validate = train_test_data_setup(dataset)
        part_of_dataset_to_compose = 0.8
    else:
        # decreasing dataset size to accelerate composing
        dataset_train_fit, dataset_to_validate = train_test_data_setup(dataset)
        dataset_to_compose, _ = train_test_data_setup(dataset_train_fit, split_ratio=part_of_dataset_to_compose/0.8)

    if not fit_fixed_chain:
        # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
        available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

        # the choice of the metric for the chain quality assessment during composition
        metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)

        # the choice and initialisation of the GP search
        composer_requirements = GPComposerRequirements(
            primary=available_model_types,
            secondary=available_model_types, max_arity=2,
            max_depth=3, pop_size=20, num_of_generations=20,
            crossover_prob=0.8, mutation_prob=0.8,
            max_lead_time=max_lead_time, add_single_model_chains=False)

        # Create GP-based composer
        composer = GPComposer()
        print(f'Dataset size to compose: {part_of_dataset_to_compose * 100}%')

        # the optimal chain generation by composition - the most time-consuming task
        chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                    initial_chain=None,
                                                    composer_requirements=composer_requirements,
                                                    metrics=metric_function,
                                                    is_visualise=is_visualise)
    else:
        chain_evo_composed = fixed_chain(list_of_nodes)

    print(f'Dataset size to fit: 80%')

    start_time = datetime.datetime.now()
    if not full_dataset:
        chain_evo_composed.fit(input_data=dataset_train_fit, verbose=True, use_cache=False)
    else:
        chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True, use_cache=False)
    time = datetime.datetime.now() - start_time
    print(f'Required time for fitting composed chain with fit method: {time}')

    # obtained chain visualisation
    if is_visualise:
        ComposerVisualiser.visualise(chain_evo_composed)

    print(f'Dataset size to validate: 20%')

    # the quality assessment for the obtained composite models
    roc_on_valid_evo_composed = RocAucMetric.get_value(chain_evo_composed, dataset_to_validate)
    acc_on_valid_evo_composed = AccuracyScore.get_value(chain_evo_composed, dataset_to_validate)
    f1_on_valid_evo_composed = F1Metric.get_value(chain_evo_composed, dataset_to_validate)
    precis_on_valid_evo_composed = PrecisionMetric.get_value(chain_evo_composed, dataset_to_validate)
    recall_on_valid_evo_composed = RecallMetric.get_value(chain_evo_composed, dataset_to_validate)
    print(f'Composed ROC AUC: {-roc_on_valid_evo_composed}')
    print(f'Composed accuracy: {-acc_on_valid_evo_composed}')
    print(f'Composed f1: {-f1_on_valid_evo_composed}')
    print(f'Composed precision: {-precis_on_valid_evo_composed}')
    print(f'Composed recall: {-recall_on_valid_evo_composed}')
    print('====================================')
    print('Process has finished')
    return chain_evo_composed


if __name__ == '__main__':
    # the dataset was obtained from https://www.kaggle.com/ugocupcic/grasping-dataset

    # dataset path definition with respect to the project root
    relative_dataset_path = 'cases/data/robotics/dataset.csv'
    full_dataset_path = os.path.join(str(project_root()), relative_dataset_path)

    # run composition
    results = run_grasp_robustness_problem(dataset_path=full_dataset_path, list_of_nodes=['rf'],
                                           part_of_dataset_to_compose=0.05, full_dataset=False,
                                           is_visualise=True, fit_fixed_chain=False,
                                           max_lead_time=datetime.timedelta(minutes=180))

    # evolution visualisation
    history_file = 'C:\\Users\\Михаил\\PycharmProjects\\tmp\\history.csv'
    evolution_visualisation(history_file)
