import itertools
import uuid
from dataclasses import dataclass
from typing import List

import numpy as np

from fedot.core.operations.factory import OperationFactory
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import Node, PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.synthetic.data import classification_dataset as synthetic_dataset, \
    gauss_quantiles_dataset as gauss_quantiles


@dataclass
class OperationTemplate:
    def __init__(self, operation_type):
        self.id = str(uuid.uuid4())
        self.operation_type = operation_type
        self.input_shape = []
        self.output_shape = []
        self.parents = []
        self.operation_instance = None
        self.fitted_operation = None
        self.data_fit = None

    def __eq__(self, other):
        return self.id == other.id


def chain_template_random(operation_types, depth, operations_per_level,
                          samples, features):
    operations_by_level = []

    for level in range(depth - 1):
        selected_operations = np.random.choice(operation_types, operations_per_level[level])
        templates = [OperationTemplate(operation_type=operation_type) for operation_type in selected_operations]
        operations_by_level.append(templates)

    root_operation = np.random.choice(operation_types)
    operations_by_level.append([OperationTemplate(operation_type=root_operation)])

    operations_by_level = with_random_links(operations_by_level)

    features_shape = [samples, features]
    # TODO: change target_shape later if non-classification problems will be used
    target_shape = [samples, 1]
    operations_by_level = with_calculated_shapes(operations_by_level,
                                                 features_shape=features_shape,
                                                 target_shape=target_shape)
    return operations_by_level


def chain_template_balanced_tree(operation_types, depth, operations_per_level,
                                 samples, features):
    operations_by_level = []

    for level in range(depth - 1):
        selected_operations = np.random.choice(operation_types, operations_per_level[level])
        templates = [OperationTemplate(operation_type=operation_type) for operation_type in selected_operations]
        operations_by_level.append(templates)

    assert operations_per_level[-1] == 1
    root_operation = np.random.choice(operation_types)
    operations_by_level.append([OperationTemplate(operation_type=root_operation)])

    operations_by_level = with_balanced_tree_links(operations_by_level)

    features_shape = [samples, features]
    # TODO: change target_shape later if non-classification problems will be used
    target_shape = [samples, 1]
    operations_by_level = with_calculated_shapes(operations_by_level,
                                                 features_shape=features_shape,
                                                 target_shape=target_shape)
    return operations_by_level


# TODO: fix internal primary nodes appearance
def with_random_links(operations_by_level):
    for current_lvl in range(len(operations_by_level) - 1):
        next_lvl = current_lvl + 1
        operations_on_current_lvl = operations_by_level[current_lvl]

        for operation in operations_on_current_lvl:
            operations_on_next_lvl = operations_by_level[next_lvl]
            links_amount = np.random.randint(1, len(operations_on_next_lvl) + 1)
            operations_to_link = np.random.choice(operations_by_level[next_lvl], links_amount, replace=False)
            for operation_ in operations_to_link:
                operation_.parents.append(operation)
    return operations_by_level


def with_balanced_tree_links(operations_by_level):
    for current_lvl in range(len(operations_by_level) - 1):
        next_lvl = current_lvl + 1
        operations_on_lvl = operations_by_level[current_lvl]

        current_level_amount = len(operations_by_level[current_lvl])
        next_level_amount = len(operations_by_level[next_lvl])
        assert current_level_amount >= next_level_amount

        operations_per_group = current_level_amount // next_level_amount

        current_group_idx = 0
        for operation_idx in range(current_level_amount):
            is_group_idx_correct = current_group_idx < next_level_amount - 1
            is_non_zero = operation_idx != 0
            if (operation_idx % operations_per_group) == 0 and is_non_zero and is_group_idx_correct:
                current_group_idx += 1
            current_operation = operations_on_lvl[operation_idx]
            current_group_parent = operations_by_level[next_lvl][current_group_idx]

            current_group_parent.parents.append(current_operation)

    return operations_by_level


def with_calculated_shapes(operations_by_level, features_shape, target_shape):
    samples_idx, classes_idx = 0, 1
    samples, classes = features_shape[samples_idx], target_shape[classes_idx]

    # Fill the first level
    for operation in operations_by_level[0]:
        operation.input_shape = features_shape
        operation.output_shape = target_shape

    for level in range(1, len(operations_by_level)):
        for operation in operations_by_level[level]:
            if len(operation.parents) == 0:
                input_features = features_shape[classes_idx]
            else:
                input_features = sum([parent.output_shape[classes_idx] for parent in operation.parents])

            operation.input_shape = [samples, input_features]

            operation.output_shape = target_shape

    return operations_by_level


def show_chain_template(operations_by_level):
    for level in range(len(operations_by_level)):
        print(f'Level = {level}')
        for operation in operations_by_level[level]:
            print(f'{operation.operation_type}, input = {operation.input_shape}, output = {operation.output_shape}')


# TODO: refactor skip_fit logic
def fit_template(chain_template, classes, with_gaussian=False, skip_fit=False):
    templates_by_operations = []
    for operation_template in itertools.chain.from_iterable(chain_template):

        # Get appropriate operation (DataOperation or Model)
        strategy_operator = OperationFactory(operation_name=operation_template.operation_type)
        operation_instance = strategy_operator.get_operation()

        operation_template.operation_instance = operation_instance
        templates_by_operations.append((operation_template, operation_instance))
    if skip_fit:
        return

    for template, instance in templates_by_operations:
        samples, features_amount = template.input_shape

        if with_gaussian:
            features, target = gauss_quantiles(samples_amount=samples,
                                               features_amount=features_amount,
                                               classes_amount=classes)
        else:
            options = {
                'informative': features_amount,
                'redundant': 0,
                'repeated': 0,
                'clusters_per_class': 1
            }
            features, target = synthetic_dataset(samples_amount=samples,
                                                 features_amount=features_amount,
                                                 classes_amount=classes,
                                                 features_options=options)
        target = np.expand_dims(target, axis=1)
        data_train = InputData(idx=np.arange(0, samples),
                               features=features, target=target,
                               data_type=DataTypesEnum.table,
                               task=Task(TaskTypesEnum.classification))

        print(f'Fit {instance}')
        fitted_operation, predictions = instance.fit(data=data_train)

        template.fitted_operation = fitted_operation
        template.data_fit = data_train


def real_chain(chain_template, with_cache=True):
    nodes_by_templates = []
    for level in range(0, len(chain_template)):
        for template in chain_template[level]:
            if len(template.parents) == 0:
                node = PrimaryNode(operation_type=template.operation_type)
            else:
                node = SecondaryNode(nodes_from=real_parents(nodes_by_templates,
                                                             template),
                                     operation_type=template.operation_type)
            node.operation = template.operation_instance
            if with_cache:
                node.fitted_model = template.fitted_operation
            nodes_by_templates.append((node, template))

    chain = Chain()
    for node, _ in nodes_by_templates:
        chain.add_node(node)

    return chain


def real_parents(nodes_by_templates, template_child) -> List[Node]:
    parent_nodes = []
    for parent in template_child.parents:
        for node, template in nodes_by_templates:
            if template == parent:
                parent_nodes.append(node)
                break
    return parent_nodes
