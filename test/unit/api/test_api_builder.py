from inspect import signature
from itertools import chain

import pytest

from fedot import Fedot, FedotBuilder
from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.api.api_utils.params import ApiParams
from fedot.core.repository.tasks import TaskTypesEnum


@pytest.fixture(name='fedot_builder_methods', scope='session')
def get_fedot_builder_methods():
    return {func_name: func for func_name in dir(FedotBuilder) if
            callable(func := getattr(FedotBuilder, func_name)) and
            not func_name.startswith('_')
            }


def test_setters_chain_returns_the_builder(fedot_builder_methods):
    builder = FedotBuilder('classification')
    for method_name in fedot_builder_methods.keys():
        if method_name in ['build']:  # add method names if needed.
            continue
        method = getattr(builder, method_name)
        builder = method()

    assert isinstance(builder, FedotBuilder)


@pytest.mark.parametrize('task_type', TaskTypesEnum)
def test_fedot_api_creation_preserves_default_params(task_type):
    if task_type is TaskTypesEnum.clustering:
        return
    task_type = task_type.name
    builder = FedotBuilder(task_type)
    fedot = builder.build()
    fedot_params = fedot.params
    default_params = ApiParams(input_params={}, problem=task_type)

    assert isinstance(fedot, Fedot)
    assert fedot_params == default_params


def test_names_and_return_annotations_of_param_setters(fedot_builder_methods):
    methods = fedot_builder_methods
    setters_by_annotation = {func_name for func_name, func in methods.items()
                             if signature(func).return_annotation == FedotBuilder.__name__}
    setters_by_name = {func_name for func_name in methods.keys() if func_name.startswith('setup_')}
    assert setters_by_annotation == setters_by_name


def test_no_unexpected_method_names(fedot_builder_methods):
    methods = fedot_builder_methods
    unexpected_method_names = {func_name for func_name in methods.keys() if not (
        func_name.startswith('setup_') or
        func_name in ['build'])}  # add method names if needed.
    assert not unexpected_method_names


def test_param_setters_has_all_api_parameters(fedot_builder_methods):
    methods = fedot_builder_methods
    setter_signs = [sign for func in methods.values()
                    if (sign := signature(func)).return_annotation == FedotBuilder.__name__]
    builder_params = set(chain(*[sign.parameters.keys() for sign in setter_signs]))
    builder_params.update(signature(FedotBuilder.__init__).parameters.keys())

    fedot_api_all_params = set(ApiParamsRepository.default_params_for_task(TaskTypesEnum.classification).keys())
    fedot_api_all_params.update(signature(Fedot.__init__).parameters.keys())
    fedot_api_all_params.discard('composer_tuner_params')  # this param is used for kwargs in Fedot.__init__

    assert builder_params == fedot_api_all_params
