import itertools
from dataclasses import dataclass

import numpy as np

from core.models.model import InputData
from core.models.model import sklearn_model_by_type
from core.repository.model_types_repository import ModelTypesIdsEnum
from experiments.generate_data import synthetic_dataset


@dataclass
class ModelTemplate:
    def __init__(self, model_type):
        self.model_type = model_type
        self.input_shape = []
        self.output_shape = []
        self.parents = []


def chain_template(model_types, depth, models_per_level,
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
        model_instance = sklearn_model_by_type(model_type=model_template.model_type)
        templates_by_models.append((model_template, model_instance))

    for template, instance in templates_by_models:
        samples, features = template.input_shape

        features, target = synthetic_dataset(samples_amount=samples,
                                             features_amount=features_amount,
                                             classes_amount=classes)
        target = np.expand_dims(target, axis=1)
        data_train = InputData(idx=np.arange(0, samples),
                               features=features, target=target)
        print(f'Fit {instance}')
        fitted_model, predictions = instance.fit(data=data_train)
        print(f'Predictions: {predictions[:10]}')


if __name__ == '__main__':
    model_types = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.knn, ModelTypesIdsEnum.xgboost]
    samples, features_amount, classes = 1000, 10, 2

    chain = chain_template(model_types=model_types, depth=4, models_per_level=3,
                           samples=samples, features=features_amount)
    show_chain_template(chain)
    fit_template(chain_template=chain, classes=classes)
