from itertools import chain
from typing import Any, Dict, List, Sequence, Type, Union

from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.optimisers.opt_history_objects.generation import Generation
from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory
from .. import any_from_json, any_to_json

MISSING_INDIVIDUAL_ARGS = {
    'metadata': {'MISSING_INDIVIDUAL': 'This individual could not be restored during `OptHistory.load()`'}
}


def _flatten_generations_list(generations_list: List[List[Individual]]) -> List[Individual]:
    def extract_intermediate_parents(ind: Individual):
        for parent in ind.parents:
            if not parent.has_native_generation:
                parents_map[parent.uid] = parent
                extract_intermediate_parents(parent)

    generations_map = {ind.uid: ind for ind in chain(*generations_list)}
    parents_map = {}
    for individual in generations_map.values():
        extract_intermediate_parents(individual)
    parents_map.update(generations_map)
    individuals_pool = list(parents_map.values())
    return individuals_pool


def _generations_list_to_uids(generations_list: List[Generation]) -> List[Generation]:
    new_list = []
    for generation in generations_list:
        new_gen = generation.copy()
        new_gen.data = [individual.uid for individual in generation]
        new_list.append(new_gen)
    return new_list


def _archive_to_uids(generations_list: List[List[Individual]]) -> List[List[str]]:
    return [[individual.uid for individual in generation] for generation in generations_list]


def opt_history_to_json(obj: OptHistory) -> Dict[str, Any]:
    serialized = any_to_json(obj)
    serialized['individuals_pool'] = _flatten_generations_list(serialized['individuals'])
    serialized['individuals'] = _generations_list_to_uids(serialized['individuals'])
    serialized['archive_history'] = _archive_to_uids(serialized['archive_history'])
    return serialized


def _uids_to_individuals(uid_sequence: Sequence[Union[str, Individual]],
                         uid_to_individual_map: Dict[str, Individual]) -> List[Individual]:
    def get_missing_individual(uid: str) -> Individual:
        individual = Individual(OptGraph(), **MISSING_INDIVIDUAL_ARGS, uid=uid)
        return individual

    def uid_to_individual_mapper(uid: Union[str, Individual]) -> Individual:
        return uid_to_individual_map.get(uid, get_missing_individual(uid)) if isinstance(uid, str) else uid

    return list(map(uid_to_individual_mapper, uid_sequence))


def _deserialize_generations_list(generations_list: List[Union[Generation, List[Union[str, Individual]]]],
                                  uid_to_individual_map: Dict[str, Individual]):
    """The operation is executed in-place"""
    for gen_num, generation in enumerate(generations_list):
        individuals = _uids_to_individuals(uid_sequence=generation,
                                           uid_to_individual_map=uid_to_individual_map)
        if isinstance(generations_list[gen_num], Generation):
            generations_list[gen_num].data = individuals
        else:
            generations_list[gen_num] = individuals


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
            if any(isinstance(i, str) for i in parent.parents):
                deserialize_intermediate_parents(parent)

    for individual in individuals:
        deserialize_intermediate_parents(individual)


def opt_history_from_json(cls: Type[OptHistory], json_obj: Dict[str, Any]) -> OptHistory:
    # backward compatibility with history._objective field
    if '_objective' in json_obj:
        json_obj['_is_multi_objective'] = json_obj['_objective'].is_multi_objective
        del json_obj['_objective']

    history = any_from_json(cls, json_obj)
    # Read all individuals from history.
    individuals_pool = history.individuals_pool
    uid_to_individual_map = {ind.uid: ind for ind in individuals_pool}
    # The attributes `individuals` and `archive_history` at the moment contain uid strings that must be converted
    # to `Individual` instances.
    _deserialize_generations_list(history.individuals, uid_to_individual_map)
    _deserialize_generations_list(history.archive_history, uid_to_individual_map)
    # Process older histories to wrap generations into the new class.
    if isinstance(history.individuals[0], list):
        history.individuals = [Generation(gen, gen_num) for gen_num, gen in enumerate(history.individuals)]
    # Deserialize parents for all generations.
    _deserialize_parent_individuals(list(chain(*history.individuals)), uid_to_individual_map)
    # The attribute is used only for serialization.
    del history.individuals_pool
    return history
