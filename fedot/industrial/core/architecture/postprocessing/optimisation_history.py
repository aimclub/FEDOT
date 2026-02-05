import pickle
from copy import copy
from typing import Any, Dict, Type

from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from golem.serializers import Serializer
from golem.serializers.any_serialization import any_from_json
from golem.serializers.coders import parent_operator_from_json


class MySerializer(Serializer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        parent_operator_binding = copy(
            Serializer.CODERS_BY_TYPE[ParentOperator])
        parent_operator_binding.update(
            {Serializer._from_json: update_parent_operator})
        Serializer.CODERS_BY_TYPE[ParentOperator] = parent_operator_binding
        individual_binding = copy(Serializer.CODERS_BY_TYPE[Individual])
        individual_binding.update({Serializer._from_json: update_individual})
        Serializer.CODERS_BY_TYPE[Individual] = individual_binding


def update_parent_operator(
        cls: Type[ParentOperator], json_obj: Dict[str, Any]):
    json_obj['type_'] = json_obj.pop('operator_type')
    json_obj['operators'] = (json_obj.pop('operator_name'),)
    deserialized = parent_operator_from_json(cls, json_obj)
    return deserialized


def update_individual(cls: Type[Individual], json_obj: Dict[str, Any]):
    deserialized = any_from_json(cls, json_obj)
    if deserialized.parent_operators:
        parents = deserialized.parent_operators[-1].parent_individuals
        operators = [deserialized.parent_operators[-1].operators[0]]
        for p_o in reversed(deserialized.parent_operators[:-1]):
            if p_o.parent_individuals != parents:
                break
            operators.append(p_o.operators[0])
        parent_operator = deserialized.parent_operators[-1]
        object.__setattr__(parent_operator, 'operators',
                           tuple(reversed(operators)))
    else:
        parent_operator = None
    object.__setattr__(deserialized, 'parent_operator', parent_operator)
    object.__delattr__(deserialized, 'parent_operators')
    return deserialized


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        renamed_module = module
        changed_import_list = ['fedot.industrial.core.repository.initializer_industrial_models']
        if module in changed_import_list:
            renamed_module = module.replace("golem.core.utilities",
                                            "fedot.industrial.core.repository.industrial_implementations.optimisation")
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()
