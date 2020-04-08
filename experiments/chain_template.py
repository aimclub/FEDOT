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

    for level in range(depth):
        selected_models = np.random.choice(model_types, models_per_level)
        templates = [ModelTemplate(model_type=model_type) for model_type in selected_models]
        models_by_level.append(templates)

    models_by_level = with_random_links(models_by_level)

    return models_by_level


# TODO: test this function
def with_random_links(models_by_level):
    for current_lvl in range(len(models_by_level) - 1):
        next_lvl = current_lvl + 1
        models_on_lvl = models_by_level[current_lvl]

        for model in models_on_lvl:
            links_amount = np.random.randint(0, len(models_by_level[next_lvl]))
            models_to_link = np.random.choice(models_by_level[next_lvl], links_amount, replace=False)
            for model_ in models_to_link:
                model_.parents.append(model)
    return models_by_level


if __name__ == '__main__':
    model_types = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.knn, ModelTypesIdsEnum.xgboost]
    chain_template(model_types=model_types, depth=3, models_per_level=2)
