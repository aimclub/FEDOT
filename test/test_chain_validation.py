import pytest

from core.chain_validation import (
    has_no_cycle,
    has_primary_nodes,
    has_no_self_cycled_nodes,
    has_no_isolated_nodes,
    validate,
    has_no_isolated_components,
)
from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.repository.model_types_repository import ModelTypesIdsEnum

ERROR_PREFIX = 'Invalid chain configuration:'


def valid_chain():
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    second = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                          nodes_from=[first])
    third = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[second])
    last = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                        nodes_from=[third])

    chain = Chain()
    for node in [first, second, third, last]:
        chain.add_node(node)

    return chain


def chain_with_cycle():
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    second = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                          nodes_from=[first])
    third = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[second, first])
    second.nodes_from.append(third)
    chain = Chain()
    for node in [first, second, third]:
        chain.add_node(node)

    return chain


def chain_with_isolated_nodes():
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    second = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                          nodes_from=[first])
    third = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[second])
    isolated = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                            nodes_from=[])
    chain = Chain()

    for node in [first, second, third, isolated]:
        chain.add_node(node)

    return chain


def chain_with_multiple_roots():
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    root_first = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                              nodes_from=[first])
    root_second = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                               nodes_from=[first])
    chain = Chain()

    for node in [first, root_first, root_second]:
        chain.add_node(node)

    return chain


def chain_with_secondary_nodes_only():
    first = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[])
    second = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                          nodes_from=[first])
    chain = Chain()
    chain.add_node(first)
    chain.add_node(second)

    return chain


def chain_with_self_cycle():
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    second = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                          nodes_from=[first])
    second.nodes_from.append(second)

    chain = Chain()
    chain.add_node(first)
    chain.add_node(second)

    return chain


def chain_with_isolated_components():
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    second = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                          nodes_from=[first])
    third = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[])
    fourth = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                          nodes_from=[third])

    chain = Chain()
    for node in [first, second, third, fourth]:
        chain.add_node(node)

    return chain


def test_chain_with_cycle_raise_exception():
    chain = chain_with_cycle()
    with pytest.raises(Exception) as exc:
        assert has_no_cycle(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has cycles'


def test_chain_without_cycles_correct():
    chain = valid_chain()

    assert has_no_cycle(chain)


def test_chain_with_isolated_nodes_raise_exception():
    chain = chain_with_isolated_nodes()
    with pytest.raises(ValueError) as exc:
        assert has_no_isolated_nodes(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has isolated nodes'


def test_multi_root_chain_raise_exception():
    chain = chain_with_multiple_roots()

    with pytest.raises(Exception) as exc:
        assert chain.root_node
    assert str(exc.value) == f'{ERROR_PREFIX} More than 1 root_nodes in chain'


def test_chain_with_primary_nodes_correct():
    chain = valid_chain()
    assert has_primary_nodes(chain)


def test_chain_without_primary_nodes_raise_exception():
    chain = chain_with_secondary_nodes_only()
    with pytest.raises(Exception) as exc:
        assert has_primary_nodes(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain does not have primary nodes'


def test_chain_with_self_cycled_nodes_raise_exception():
    chain = chain_with_self_cycle()
    with pytest.raises(Exception) as exc:
        assert has_no_self_cycled_nodes(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has self-cycled nodes'


def test_chain_validate_correct():
    chain = valid_chain()
    validate(chain)


def test_chain_with_isolated_components_raise_exception():
    chain = chain_with_isolated_components()
    with pytest.raises(Exception) as exc:
        assert has_no_isolated_components(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has isolated components'
