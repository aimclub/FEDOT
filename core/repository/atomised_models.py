import os
from dataclasses import dataclass
from pickle import dump

from aenum import extend_enum

from core.repository.model_types_repository import ModelTypesIdsEnum

_atomised_models_prefix = 'atomised_'


def _atomised_models_folder():
    path = os.path.join(str(os.path.dirname(__file__)), 'atomised_models')
    return path


def atomise_chain(model_name: str, chain: 'Chain'):
    path_to_file = f'{_atomised_models_folder()}/{model_name}.pkl'
    with open(path_to_file, 'wb') as pickle_file:
        dump(chain, pickle_file)
    return read_atomised_model(model_name)


def read_atomised_model(model_name: str):
    path_to_file = f'{_atomised_models_folder()}/{model_name}.pkl'
    new_enum_name = f'{_atomised_models_prefix}{model_name}'
    extend_enum(ModelTypesIdsEnum, new_enum_name, AtomisedModelDescription(path_to_file=path_to_file))
    return ModelTypesIdsEnum[new_enum_name]


def is_model_atomised(model_type: ModelTypesIdsEnum):
    return model_type.name.startswith(_atomised_models_prefix)


@dataclass
class AtomisedModelDescription:
    path_to_file: str
