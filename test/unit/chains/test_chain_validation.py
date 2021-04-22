import pytest

from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_validation import (has_correct_operation_positions, has_no_cycle,
                                                has_no_isolated_components, has_no_isolated_nodes,
                                                has_no_self_cycled_nodes, has_primary_nodes,
                                                validate, has_final_operation_as_model,
                                                has_no_conflicts_with_data_flow,
                                                is_chain_contains_ts_operations,
                                                has_no_data_flow_conflicts_in_ts_chain,
                                                only_ts_specific_operations_are_primary)
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.repository.tasks import Task, TaskTypesEnum

ERROR_PREFIX = 'Invalid chain configuration:'


def valid_chain():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit',
                           nodes_from=[first])
    third = SecondaryNode(operation_type='logit',
                          nodes_from=[second])
    last = SecondaryNode(operation_type='logit',
                         nodes_from=[third])

    chain = Chain()
    for node in [first, second, third, last]:
        chain.add_node(node)

    return chain


def chain_with_cycle():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit',
                           nodes_from=[first])
    third = SecondaryNode(operation_type='logit',
                          nodes_from=[second, first])
    second.nodes_from.append(third)
    chain = Chain()
    for node in [first, second, third]:
        chain.add_node(node)

    return chain


def chain_with_isolated_nodes():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit',
                           nodes_from=[first])
    third = SecondaryNode(operation_type='logit',
                          nodes_from=[second])
    isolated = SecondaryNode(operation_type='logit',
                             nodes_from=[])
    chain = Chain()

    for node in [first, second, third, isolated]:
        chain.add_node(node)

    return chain


def chain_with_multiple_roots():
    first = PrimaryNode(operation_type='logit')
    root_first = SecondaryNode(operation_type='logit',
                               nodes_from=[first])
    root_second = SecondaryNode(operation_type='logit',
                                nodes_from=[first])
    chain = Chain()

    for node in [first, root_first, root_second]:
        chain.add_node(node)

    return chain


def chain_with_secondary_nodes_only():
    first = SecondaryNode(operation_type='logit',
                          nodes_from=[])
    second = SecondaryNode(operation_type='logit',
                           nodes_from=[first])
    chain = Chain()
    chain.add_node(first)
    chain.add_node(second)

    return chain


def chain_with_self_cycle():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit',
                           nodes_from=[first])
    second.nodes_from.append(second)

    chain = Chain()
    chain.add_node(first)
    chain.add_node(second)

    return chain


def chain_with_isolated_components():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit',
                           nodes_from=[first])
    third = SecondaryNode(operation_type='logit',
                          nodes_from=[])
    fourth = SecondaryNode(operation_type='logit',
                           nodes_from=[third])

    chain = Chain()
    for node in [first, second, third, fourth]:
        chain.add_node(node)

    return chain


def chain_with_incorrect_root_operation():
    first = PrimaryNode(operation_type='logit')
    second = PrimaryNode(operation_type='logit')
    final = SecondaryNode(operation_type='scaling',
                          nodes_from=[first, second])

    chain = Chain(final)

    return chain


def chain_with_incorrect_task_type():
    first = PrimaryNode(operation_type='linear')
    second = PrimaryNode(operation_type='linear')
    final = SecondaryNode(operation_type='kmeans',
                          nodes_from=[first, second])

    chain = Chain(final)

    return chain, Task(TaskTypesEnum.classification)


def chain_with_only_data_operations():
    first = PrimaryNode(operation_type='one_hot_encoding')
    second = SecondaryNode(operation_type='scaling', nodes_from=[first])
    final = SecondaryNode(operation_type='ransac_lin_reg', nodes_from=[second])

    chain = Chain(final)

    return chain


def chain_with_incorrect_data_flow():
    """ When combining the features in the presented chain, a table with 5
    columns will turn into a table with 10 columns """
    first = PrimaryNode(operation_type='scaling')
    second = PrimaryNode(operation_type='ransac_lin_reg')

    final = SecondaryNode(operation_type='ridge', nodes_from=[first, second])
    chain = Chain(final)
    return chain


def ts_chain_with_incorrect_data_flow():
    """
    Connection lagged -> lagged is incorrect
    Connection ridge -> ar is incorrect also
       lagged - lagged - ridge \
                                ar -> final forecast
                lagged - ridge /
    """

    # First level
    node_lagged = PrimaryNode('lagged')

    # Second level
    node_lagged_1 = SecondaryNode('lagged', nodes_from=[node_lagged])
    node_lagged_2 = PrimaryNode('lagged')

    # Third level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    # Fourth level - root node
    node_final = SecondaryNode('ar', nodes_from=[node_ridge_1, node_ridge_2])
    chain = Chain(node_final)

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


def test_chain_with_incorrect_task_type_raise_exception():
    chain, task = chain_with_incorrect_task_type()
    with pytest.raises(Exception) as exc:
        assert has_correct_operation_positions(chain, task)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has incorrect operations positions'


def test_chain_without_model_in_root_node():
    incorrect_chain = chain_with_only_data_operations()

    with pytest.raises(Exception) as exc:
        assert has_final_operation_as_model(incorrect_chain)

    assert str(exc.value) == f'{ERROR_PREFIX} Root operation is not a model'


def test_chain_with_incorrect_data_flow():
    incorrect_chain = chain_with_incorrect_data_flow()

    with pytest.raises(Exception) as exc:
        assert has_no_conflicts_with_data_flow(incorrect_chain)

    assert str(exc.value) == f'{ERROR_PREFIX} Chain has incorrect subgraph with wrong parent nodes combination'


def test_ts_chain_with_incorrect_data_flow():
    incorrect_chain = ts_chain_with_incorrect_data_flow()

    if is_chain_contains_ts_operations(incorrect_chain):
        with pytest.raises(Exception) as exc:
            assert has_no_data_flow_conflicts_in_ts_chain(incorrect_chain)

        assert str(exc.value) == f'{ERROR_PREFIX} Chain has incorrect subgraph with wrong parent nodes combination'
    else:
        assert False


def test_only_ts_specific_operations_are_primary():
    """ Incorrect chain
    lagged \
             linear -> final forecast
     ridge /
    """
    node_lagged = PrimaryNode('lagged')
    node_ridge = PrimaryNode('ridge')
    node_final = SecondaryNode('linear', nodes_from=[node_lagged, node_ridge])
    incorrect_chain = Chain(node_final)

    with pytest.raises(Exception) as exc:
        assert only_ts_specific_operations_are_primary(incorrect_chain)

    assert str(exc.value) == f'{ERROR_PREFIX} Chain for forecasting has not ts_specific preprocessing in primary nodes'
