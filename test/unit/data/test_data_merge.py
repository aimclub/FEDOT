import numpy as np

from examples.regression_with_tuning_example import get_regression_dataset
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.merge import DataMerger, TaskTargetMerger
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

np.random.seed(2021)


def generate_outputs():
    """ Function for simple case with non-equal outputs in list """
    idx_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    idx_2 = [2, 3, 4, 5, 6, 7, 8, 9]

    task = Task(TaskTypesEnum.regression)
    generated_target = np.random.sample((len(idx_1), 1))
    generated_features = np.random.sample((len(idx_1), 2))

    list_with_outputs = []
    for idx, data_flow_len in zip([idx_1, idx_2], [1, 0]):
        data_info = SupplementaryData(data_flow_length=data_flow_len)
        output_data = OutputData(idx=idx,
                                 features=generated_features[idx, :],
                                 predict=generated_target[idx, :],
                                 task=task,
                                 target=generated_target[idx, :],
                                 data_type=DataTypesEnum.table,
                                 supplementary_data=data_info)
        list_with_outputs.append(output_data)

    return list_with_outputs, idx_1, idx_2


def test_data_merge_in_pipeline():
    """ Test check is the pipeline can correctly work with dynamic changes in
    tables during the fit process
    """

    #   ridge
    #  /     \ (merge operation)
    # |   ransac_lin_reg (remove several lines in table)
    #  \     /
    #  scaling

    node_scaling = PrimaryNode('scaling')

    node_lin_ransac = SecondaryNode('ransac_lin_reg', nodes_from=[node_scaling])
    node_final = SecondaryNode('ridge', nodes_from=[node_lin_ransac, node_scaling])
    pipeline = Pipeline(node_final)

    features_options = {'informative': 2, 'bias': 2.0}
    x_train, y_train, x_test, y_test = get_regression_dataset(features_options=features_options,
                                                              samples_amount=100,
                                                              features_amount=5)
    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    # Fit and predict
    pipeline.fit_from_scratch(train_input)
    prediction = pipeline.predict(train_input)

    assert prediction is not None


def test_data_merge_function():
    """ Test check is the merge function can find appropriate intersections of
    indices. Set {idx_2} ∈ set {idx_1}, so intersection must be = idx_2
    """

    list_with_outputs, idx_1, idx_2 = generate_outputs()

    new_idx, features, target, task, d_type, updated_info = DataMerger(list_with_outputs).merge()

    assert tuple(new_idx) == tuple(idx_2)


def test_target_task_two_ignore_merge():
    """ The test runs an example of how different targets and tasks will be
    combined. Consider situation when one target should be untouched"""

    # Targets in different outputs
    labels_col = [[1], [1]]
    probabilities_col_1 = [[0.8], [0.7]]
    probabilities_col_2 = [[0.5], [0.5]]
    targets = np.array([labels_col,
                        probabilities_col_1,
                        probabilities_col_2])

    # Flags for targets
    main_targets = [True, False, False]

    # Tasks
    class_task = Task(TaskTypesEnum.classification)
    regr_task = Task(TaskTypesEnum.classification)
    tasks = [class_task, regr_task, regr_task]

    merger = TaskTargetMerger(None)
    target, is_main_target, task = merger.ignored_merge(targets, main_targets, tasks)

    assert is_main_target is True
    assert task.task_type is TaskTypesEnum.classification


def test_target_task_two_none_merge():
    """ The test runs an example of how different targets and tasks will be
    combined. Consider situation when two targets are main ones (labeled as None)
    """

    # Targets in different outputs
    labels_col = [[1], [1]]
    labels_col_copy = [[1], [1]]
    probabilities_col = [[0.5], [0.5]]
    targets = np.array([labels_col,
                        labels_col_copy,
                        probabilities_col])

    # Flags for targets
    main_targets = [True, True, False]

    # Tasks
    class_task = Task(TaskTypesEnum.classification)
    regr_task = Task(TaskTypesEnum.classification)
    tasks = [class_task, class_task, regr_task]

    merger = TaskTargetMerger(None)
    target, is_main_target, task = merger.ignored_merge(targets, main_targets, tasks)

    assert is_main_target is True
    assert task.task_type is TaskTypesEnum.classification
