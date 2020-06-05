import random
from typing import List

from core.composer.chain import Chain
from core.repository.model_types_repository import ModelTypesIdsEnum
from utilities.synthetic.chain_template import (
    chain_template_random,
    fit_template,
    real_chain
)


def chain_with_fixed_structure() -> Chain:
    """
    Generates chain with a fixed structure of nodes and links
    :return:
    """
    raise NotImplementedError()


def chain_with_random_links(depth: int, models_per_level: List[int],
                            used_models: List[ModelTypesIdsEnum]) -> Chain:
    """
    Generates chain with a fixed structure of nodes but random links.
    :param depth: Tree depth.
    :param models_per_level: The amount of models at each layer.
    :param used_models: The list of models to be randomly included into
    the resulted chain.
    :return: Chain with random links.
    """
    template = chain_template_random(model_types=used_models,
                                     depth=depth, models_per_level=models_per_level,
                                     samples=100, features=10)
    fit_template(chain_template=template, classes=2, skip_fit=True)
    resulted_chain = real_chain(chain_template=template)

    return resulted_chain


def chain_full_random(depth: int, max_level_size,
                      used_models: List[ModelTypesIdsEnum]) -> Chain:
    """
    Generates chain with random amount of nodes and links.
    :param depth: Tree depth.
    :param max_level_size: Max amount of nodes per level.
    :param used_models: The list of models to be randomly included into
    the resulted chain.
    :return: Chain with random models_per_level and links.
    """
    models_per_lvl = _random_models_per_lvl(depth=depth,
                                            max_level_size=max_level_size)
    resulted_chain = chain_with_random_links(depth=depth,
                                             models_per_level=models_per_lvl,
                                             used_models=used_models)
    return resulted_chain


def chain_balanced_tree() -> Chain:
    """
    Generates chain with balanced tree-like structure
    :return:
    """
    raise NotImplementedError()


def _random_models_per_lvl(depth, max_level_size):
    models_per_level = []
    for _ in range(depth - 1):
        models_per_level.append(random.randint(1, max_level_size))
    last_level_size = 1
    models_per_level.append(last_level_size)

    return models_per_level


if __name__ == '__main__':
    chain = chain_with_random_links(depth=3, models_per_level=[3, 2, 1],
                                    used_models=[ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.logit])
    full_random_chain = chain_full_random(depth=4, max_level_size=4,
                                          used_models=[ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.logit])
