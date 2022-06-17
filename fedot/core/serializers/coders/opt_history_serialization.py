import operator
from copy import deepcopy
from functools import reduce
from typing import Any, Collection, Dict, List, Type, Union

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import OptHistory

from . import any_from_json, any_to_json


def _get_individuals_pool_from_generations_list(generations_list: List[List[Individual]]) -> List[Individual]:
    uid_to_individual_map = {ind.uid: ind for gen in generations_list for ind in gen}
    parents_map = {}
    for individual in uid_to_individual_map.values():
        for parent_operator in individual.parent_operators:
            for parent_ind in parent_operator.parent_individuals:
                parents_map[parent_ind.uid] = parent_ind
    uid_to_individual_map.update(parents_map)
    individuals_pool = list(uid_to_individual_map.values())
    return individuals_pool


def _serialize_generations_list(generations_list: List[List[Individual]]):
    new_generations_list = []
    for generation in generations_list:
        new_generations_list.append([individual.uid for individual in generation])
    return new_generations_list


def opt_history_to_json(obj: OptHistory) -> Dict[str, Any]:
    serialized = any_to_json(obj)
    serialized['individuals_pool'] = _get_individuals_pool_from_generations_list(serialized['individuals'])
    serialized['individuals'] = _serialize_generations_list(serialized['individuals'])
    serialized['archive_history'] = _serialize_generations_list(serialized['archive_history'])
    return serialized


def _get_individuals_from_uid_list(uid_list: List[Union[str, Individual]],
                                   uid_to_individual_map: Dict[str, Individual]) -> List[Individual]:
    return [uid_to_individual_map.get(uid, Individual(OptGraph())) if isinstance(uid, str) else uid for uid in uid_list]


def _deserialize_generations_list(generations_list: List[List[Union[str, Individual]]],
                                  uid_to_individual_map: Dict[str, Individual]):
    """The operation is executed in-place"""
    for gen_num, generation in enumerate(generations_list):
        generations_list[gen_num] = _get_individuals_from_uid_list(uid_list=generation,
                                                                   uid_to_individual_map=uid_to_individual_map)


def _deserialize_parent_individuals(generations_list: List[List[Individual]],
                                    uid_to_individual_map: Dict[str, Individual]):
    """The operation is executed in-place"""
    for gen in generations_list:
        for individual in gen:
            for parent_op in individual.parent_operators:
                parent_op.parent_individuals = \
                    _get_individuals_from_uid_list(uid_list=parent_op.parent_individuals,
                                                   uid_to_individual_map=uid_to_individual_map)


def opt_history_from_json(cls: Type[OptHistory], json_obj: Dict[str, Any]) -> OptHistory:
    deserialized = any_from_json(cls, json_obj)
    if hasattr(deserialized, 'individuals_pool'):
        # if the history has `individuals_pool`, then other attributes contain uid strings that should be converted
        # to `Individual` instances.
        uid_to_individual_map = {ind.uid: ind for ind in deserialized.individuals_pool}
        del deserialized.individuals_pool
        _deserialize_generations_list(deserialized.individuals, uid_to_individual_map)
        _deserialize_generations_list(deserialized.archive_history, uid_to_individual_map)
    else:
        # Older histories has no `individuals_pool`, and all individuals are represented in `individuals` attribute.
        # Let's gather them through all the generations.
        uid_to_individual_map = {ind.uid: ind for gen in deserialized.individuals for ind in gen}
    # Now the history attributes are guaranteed to contain individuals.
    # But parent_individuals still contain uid strings.
    _deserialize_parent_individuals(deserialized.individuals, uid_to_individual_map)
    _deserialize_parent_individuals(deserialized.archive_history, uid_to_individual_map)
    return deserialized
