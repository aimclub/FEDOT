from itertools import chain
from typing import Any, Dict, List, Type, Union

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import OptHistory
from . import any_from_json, any_to_json

MISSING_INDIVIDUAL_ARGS = {
    'graph': OptGraph(),
    'metadata': {'MISSING_INDIVIDUAL': 'This individual could not be restored during `OptHistory.load()`'}
}


def _get_individuals_pool_from_generations_list(generations_list: List[List[Individual]]) -> List[Individual]:
    uid_to_individual_map = {ind.uid: ind for ind in chain.from_iterable(generations_list)}
    parents_map = {}
    for individual in uid_to_individual_map.values():
        for parent_operator in individual.parent_operators:
            for parent_ind in parent_operator.parent_individuals:
                if parent_ind.uid in uid_to_individual_map:
                    continue
                parents_map[parent_ind.uid] = parent_ind
    uid_to_individual_map.update(parents_map)
    individuals_pool = list(uid_to_individual_map.values())
    return individuals_pool


def _serialize_generations_list(generations_list: List[List[Individual]]) -> List[List[Individual]]:
    return [[individual.uid for individual in generation] for generation in generations_list]


def opt_history_to_json(obj: OptHistory) -> Dict[str, Any]:
    serialized = any_to_json(obj)
    serialized['individuals_pool'] = _get_individuals_pool_from_generations_list(serialized['individuals'])
    serialized['individuals'] = _serialize_generations_list(serialized['individuals'])
    serialized['archive_history'] = _serialize_generations_list(serialized['archive_history'])
    return serialized


def _get_individuals_from_uid_list(uid_list: List[Union[str, Individual]],
                                   uid_to_individual_map: Dict[str, Individual]) -> List[Individual]:

    def get_missing_individual(uid: str) -> Individual:
        individual = Individual(**MISSING_INDIVIDUAL_ARGS)
        individual.uid = uid
        return individual

    def uid_to_individual_mapper(uid: Union[str, Individual]) -> Individual:
        return uid_to_individual_map.get(uid, get_missing_individual(uid)) if isinstance(uid, str) else uid

    return list(map(uid_to_individual_mapper, uid_list))


def _deserialize_generations_list(generations_list: List[List[Union[str, Individual]]],
                                  uid_to_individual_map: Dict[str, Individual]):
    """The operation is executed in-place"""
    for gen_num, generation in enumerate(generations_list):
        generations_list[gen_num] = _get_individuals_from_uid_list(uid_list=generation,
                                                                   uid_to_individual_map=uid_to_individual_map)


def _deserialize_parent_individuals(individuals: List[Individual],
                                    uid_to_individual_map: Dict[str, Individual]):
    """The operation is executed in-place"""
    for individual in individuals:
        for parent_op in individual.parent_operators:
            parent_op.parent_individuals = \
                _get_individuals_from_uid_list(uid_list=parent_op.parent_individuals,
                                               uid_to_individual_map=uid_to_individual_map)


def opt_history_from_json(cls: Type[OptHistory], json_obj: Dict[str, Any]) -> OptHistory:
    deserialized = any_from_json(cls, json_obj)
    # Define all history individuals pool
    if hasattr(deserialized, 'individuals_pool'):
        individuals_pool = deserialized.individuals_pool
    else:
        # Older histories has no `individuals_pool`, and all individuals are represented in `individuals` attribute.
        # Let's gather them through all the generations.
        individuals_pool = list(chain.from_iterable(deserialized.individuals))
    # Deserialize parents for all pipelines
    uid_to_individual_map = {ind.uid: ind for ind in individuals_pool}
    _deserialize_parent_individuals(individuals_pool, uid_to_individual_map)
    # if the history has `individuals_pool`, then other attributes contain uid strings that should be converted
    # to `Individual` instances.
    if hasattr(deserialized, 'individuals_pool'):
        _deserialize_generations_list(deserialized.individuals, uid_to_individual_map)
        _deserialize_generations_list(deserialized.archive_history, uid_to_individual_map)
        del deserialized.individuals_pool

    return deserialized
