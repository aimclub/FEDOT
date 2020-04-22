import itertools
import uuid
from dataclasses import dataclass

import numpy as np

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode, FittedModelCache
from core.models.model import InputData
from core.models.model import Model
from core.repository.task_types import MachineLearningTasksEnum
from experiments.generate_data import synthetic_dataset


@dataclass
class ModelTemplate:
    def __init__(self, model_type):
        self.id = str(uuid.uuid4())
        self.model_type = model_type
        self.input_shape = []
        self.output_shape = []
        self.parents = []
        self.model_instance = None
        self.fitted_model = None

    def __eq__(self, other):
        return self.id == other.id


def chain_template_random(model_types, depth, models_per_level,
                          samples, features):
    models_by_level = []

    for level in range(depth - 1):
        selected_models = np.random.choice(model_types, models_per_level)
        templates = [ModelTemplate(model_type=model_type) for model_type in selected_models]
        models_by_level.append(templates)

    last_model = np.random.choice(model_types)
    models_by_level.append([ModelTemplate(model_type=last_model)])

    models_by_level = with_random_links(models_by_level)

    features_shape = [samples, features]
    # TODO: change target_shape later if non-classification problems will be used
    target_shape = [samples, 1]
    models_by_level = with_calculated_shapes(models_by_level,
                                             source_features=features_shape,
                                             source_target=target_shape)
    return models_by_level


def chain_template_balanced_tree(model_types, depth, models_per_level,
                                 samples, features):
    models_by_level = []

    for level in range(depth - 1):
        selected_models = np.random.choice(model_types, models_per_level[level])
        templates = [ModelTemplate(model_type=model_type) for model_type in selected_models]
        models_by_level.append(templates)

    assert models_per_level[-1] == 1
    last_model = np.random.choice(model_types)
    models_by_level.append([ModelTemplate(model_type=last_model)])

    models_by_level = with_balanced_tree_links(models_by_level)

    features_shape = [samples, features]
    # TODO: change target_shape later if non-classification problems will be used
    target_shape = [samples, 1]
    models_by_level = with_calculated_shapes(models_by_level,
                                             source_features=features_shape,
                                             source_target=target_shape)
    return models_by_level


# TODO: fix internal primary nodes appearance
def with_random_links(models_by_level):
    for current_lvl in range(len(models_by_level) - 1):
        next_lvl = current_lvl + 1
        models_on_lvl = models_by_level[current_lvl]

        for model in models_on_lvl:
            links_amount = np.random.randint(1, len(models_by_level[next_lvl]) + 1)
            models_to_link = np.random.choice(models_by_level[next_lvl], links_amount, replace=False)
            for model_ in models_to_link:
                model_.parents.append(model)
    return models_by_level


def with_balanced_tree_links(models_by_level):
    for current_lvl in range(len(models_by_level) - 1):
        next_lvl = current_lvl + 1
        models_on_lvl = models_by_level[current_lvl]

        current_level_amount = len(models_by_level[current_lvl])
        next_level_amount = len(models_by_level[next_lvl])
        assert current_level_amount >= next_level_amount

        models_per_group = current_level_amount // next_level_amount

        current_group_idx = 0
        for model_idx in range(current_level_amount):
            if (model_idx % models_per_group) == 0 and model_idx != 0 and current_group_idx < next_level_amount - 1:
                current_group_idx += 1
            current_model = models_on_lvl[model_idx]
            current_group_parent = models_by_level[next_lvl][current_group_idx]

            current_group_parent.parents.append(current_model)

    return models_by_level


def with_calculated_shapes(models_by_level, source_features, source_target):
    samples, classes = source_features[0], source_target[1]

    # Fill the first level
    for model in models_by_level[0]:
        model.input_shape = source_features
        model.output_shape = source_target

    for level in range(1, len(models_by_level)):
        for model in models_by_level[level]:
            if len(model.parents) == 0:
                input_features = source_features[1]
            else:
                input_features = sum([parent.output_shape[1] for parent in model.parents])

            model.input_shape = [samples, input_features]

            model.output_shape = source_target

    return models_by_level


def show_chain_template(models_by_level):
    for level in range(len(models_by_level)):
        print(f'Level = {level}')
        for model in models_by_level[level]:
            print(f'{model.model_type}, input = {model.input_shape}, output = {model.output_shape}')


def fit_template(chain_template, classes):
    templates_by_models = []
    for model_template in itertools.chain.from_iterable(chain_template):
        model_instance = Model(model_type=model_template.model_type)
        templates_by_models.append((model_template, model_instance))

    for template, instance in templates_by_models:
        samples, features_amount = template.input_shape

        features, target = synthetic_dataset(samples_amount=samples,
                                             features_amount=features_amount,
                                             classes_amount=classes)
        target = np.expand_dims(target, axis=1)
        data_train = InputData(idx=np.arange(0, samples),
                               features=features, target=target,
                               task_type=MachineLearningTasksEnum.classification)
        print(f'Fit {instance}')
        fitted_model, predictions = instance.fit(data=data_train)

        template.model_instance = instance
        template.fitted_model = fitted_model


def real_chain(chain_template):
    nodes_by_templates = []
    for level in range(0, len(chain_template)):
        for template in chain_template[level]:
            if len(template.parents) == 0:
                node = PrimaryNode(model_type=template.model_type)
            else:
                node = SecondaryNode(nodes_from=real_parents(nodes_by_templates,
                                                             template),
                                     model_type=template.model_type)
            node.model = template.model_instance
            cache = FittedModelCache(related_node=node)
            cache.append(fitted_model=template.fitted_model)
            node.cache = cache
            nodes_by_templates.append((node, template))

    chain = Chain()
    for node, _ in nodes_by_templates:
        chain.add_node(node)

    return chain


def real_parents(nodes_by_templates, template_child):
    parent_nodes = []
    for parent in template_child.parents:
        for node, template in nodes_by_templates:
            if template == parent:
                parent_nodes.append(node)
                break
    return parent_nodes
