from itertools import chain
from typing import Any, Dict, List, Sequence, Type, Union

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import OptHistory
from . import any_from_json, any_to_json

MISSING_INDIVIDUAL_ARGS = {
    'metadata': {'MISSING_INDIVIDUAL': 'This individual could not be restored during `OptHistory.load()`'}
}


def _flatten_generations_list(generations_list: List[List[Individual]]) -> List[Individual]:
    def extract_intermediate_parents(ind: Individual):
        if not ind.parent_operator:
            return
        parent_individuals = ind.parent_operator.parent_individuals
        for i in parent_individuals:
            if i.native_generation is None:
                parents_map[i.uid] = i
                extract_intermediate_parents(i)

    uid_to_individual_map = {ind.uid: ind for ind in reversed(list(chain(*generations_list)))}
    parents_map = {}
    for individual in uid_to_individual_map.values():
        extract_intermediate_parents(individual)
    uid_to_individual_map.update(parents_map)
    individuals_pool = list(uid_to_individual_map.values())
    return individuals_pool


def _generations_list_to_uids(generations_list: List[List[Individual]]) -> List[List[str]]:
    return [[individual.uid for individual in generation] for generation in generations_list]


def opt_history_to_json(obj: OptHistory) -> Dict[str, Any]:
    serialized = any_to_json(obj)
    serialized['individuals_pool'] = _flatten_generations_list(serialized['individuals'])
    serialized['individuals'] = _generations_list_to_uids(serialized['individuals'])
    serialized['archive_history'] = _generations_list_to_uids(serialized['archive_history'])
    return serialized


def _set_native_generations(generations_list: List[List[Individual]]):
    """The operation is executed in-place"""
    for gen_num, gen in enumerate(generations_list):
        for ind in gen:
            ind.set_native_generation(gen_num)


def _uids_to_individuals(uid_sequence: Sequence[Union[str, Individual]],
                         uid_to_individual_map: Dict[str, Individual]) -> List[Individual]:
    def get_missing_individual(uid: str) -> Individual:
        individual = Individual(OptGraph(), **MISSING_INDIVIDUAL_ARGS, uid=uid)
        return individual

    def uid_to_individual_mapper(uid: Union[str, Individual]) -> Individual:
        return uid_to_individual_map.get(uid, get_missing_individual(uid)) if isinstance(uid, str) else uid

    return list(map(uid_to_individual_mapper, uid_sequence))


def _deserialize_generations_list(generations_list: List[List[Union[str, Individual]]],
                                  uid_to_individual_map: Dict[str, Individual]):
    """The operation is executed in-place"""
    for gen_num, generation in enumerate(generations_list):
        generations_list[gen_num] = _uids_to_individuals(uid_sequence=generation,
                                                         uid_to_individual_map=uid_to_individual_map)


def _deserialize_parent_individuals(individuals: List[Individual],
                                    uid_to_individual_map: Dict[str, Individual]):
    """The operation is executed in-place"""
    def deserialize_intermediate_parents(ind):
        parent_op = ind.parent_operator
        if not parent_op:
            return
        parent_individuals = _uids_to_individuals(uid_sequence=parent_op.parent_individuals,
                                                  uid_to_individual_map=uid_to_individual_map)
        object.__setattr__(parent_op, 'parent_individuals', tuple(parent_individuals))
        for parent in parent_individuals:
            if parent.parent_operator and any(isinstance(i, str) for i in parent.parent_operator.parent_individuals):
                deserialize_intermediate_parents(parent)

    for individual in individuals:
        deserialize_intermediate_parents(individual)


def opt_history_from_json(cls: Type[OptHistory], json_obj: Dict[str, Any]) -> OptHistory:
    deserialized = any_from_json(cls, json_obj)
    # Define all history individuals pool
    if hasattr(deserialized, 'individuals_pool'):
        individuals_pool = deserialized.individuals_pool
    else:
        # Older histories has no `individuals_pool`, and all individuals are represented in `individuals` attribute.
        # Let's gather them through all the generations.
        _set_native_generations(deserialized.individuals)
        individuals_pool = list(chain(*deserialized.individuals))
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
