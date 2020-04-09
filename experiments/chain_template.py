from dataclasses import dataclass

import numpy as np

from core.repository.model_types_repository import ModelTypesIdsEnum


@dataclass
class ModelTemplate:
    def __init__(self, model_type):
        self.model_type = model_type
        self.input_shape = []
        self.output_shape = []
        self.parents = []


def chain_template(model_types, depth, models_per_level):
    models_by_level = []

    for level in range(depth - 1):
        selected_models = np.random.choice(model_types, models_per_level)
        templates = [ModelTemplate(model_type=model_type) for model_type in selected_models]
        models_by_level.append(templates)

    last_model = np.random.choice(model_types)
    models_by_level.append([ModelTemplate(model_type=last_model)])

    models_by_level = with_random_links(models_by_level)
    models_by_level = with_calculated_shapes(models_by_level,
                                             source_features=[100, 10],
                                             source_target=[100, 1])
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


def print_chain_template(models_by_level):
    for level in range(len(models_by_level)):
        print(f'Level = {level}')
        for model in models_by_level[level]:
            print(f'{model.model_type}, input = {model.input_shape}, output = {model.output_shape}')


if __name__ == '__main__':
    model_types = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.knn, ModelTypesIdsEnum.xgboost]
    chain = chain_template(model_types=model_types, depth=4, models_per_level=3)
    print_chain_template(chain)
