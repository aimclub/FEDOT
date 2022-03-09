import pytest

from fedot.core.dag.validation_rules import has_no_cycle, has_no_isolated_components, has_no_isolated_nodes, \
    has_no_self_cycled_nodes
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.validation import (validate)
from fedot.core.pipelines.validation_rules import has_correct_operation_positions, has_final_operation_as_model, \
    has_no_conflicts_in_decompose, has_no_conflicts_with_data_flow, has_no_data_flow_conflicts_in_ts_pipeline, \
    has_primary_nodes, is_pipeline_contains_ts_operations, only_non_lagged_operations_are_primary, \
    has_correct_data_sources, has_parent_contain_single_resample, has_no_conflicts_during_multitask, \
    has_no_conflicts_after_class_decompose
from fedot.core.repository.tasks import Task, TaskTypesEnum

PIPELINE_ERROR_PREFIX = 'Invalid pipeline configuration:'
GRAPH_ERROR_PREFIX = 'Invalid graph configuration:'


def valid_pipeline():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit',
                           nodes_from=[first])
    third = SecondaryNode(operation_type='logit',
                          nodes_from=[second])
    last = SecondaryNode(operation_type='logit',
                         nodes_from=[third])

    pipeline = Pipeline()
    for node in [first, second, third, last]:
        pipeline.add_node(node)

    return pipeline


def pipeline_with_cycle():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit',
                           nodes_from=[first])
    third = SecondaryNode(operation_type='logit',
                          nodes_from=[second, first])
    second.nodes_from.append(third)
    pipeline = Pipeline()
    for node in [first, second, third]:
        pipeline.add_node(node)

    return pipeline


def pipeline_with_isolated_nodes():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit',
                           nodes_from=[first])
    third = SecondaryNode(operation_type='logit',
                          nodes_from=[second])
    isolated = SecondaryNode(operation_type='logit',
                             nodes_from=[])
    pipeline = Pipeline()

    for node in [first, second, third, isolated]:
        pipeline.add_node(node)

    return pipeline


def pipeline_with_multiple_roots():
    first = PrimaryNode(operation_type='logit')
    root_first = SecondaryNode(operation_type='logit',
                               nodes_from=[first])
    root_second = SecondaryNode(operation_type='logit',
                                nodes_from=[first])
    pipeline = Pipeline()

    for node in [first, root_first, root_second]:
        pipeline.add_node(node)

    return pipeline


def pipeline_with_secondary_nodes_only():
    first = SecondaryNode(operation_type='logit',
                          nodes_from=[])
    second = SecondaryNode(operation_type='logit',
                           nodes_from=[first])
    pipeline = Pipeline()
    pipeline.add_node(first)
    pipeline.add_node(second)

    return pipeline


def pipeline_with_self_cycle():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit',
                           nodes_from=[first])
    second.nodes_from.append(second)

    pipeline = Pipeline()
    pipeline.add_node(first)
    pipeline.add_node(second)

    return pipeline


def pipeline_with_isolated_components():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit',
                           nodes_from=[first])
    third = SecondaryNode(operation_type='logit',
                          nodes_from=[])
    fourth = SecondaryNode(operation_type='logit',
                           nodes_from=[third])

    pipeline = Pipeline()
    for node in [first, second, third, fourth]:
        pipeline.add_node(node)

    return pipeline


def pipeline_with_incorrect_root_operation():
    first = PrimaryNode(operation_type='logit')
    second = PrimaryNode(operation_type='logit')
    final = SecondaryNode(operation_type='scaling',
                          nodes_from=[first, second])

    pipeline = Pipeline(final)

    return pipeline


def pipeline_with_incorrect_task_type():
    first = PrimaryNode(operation_type='linear')
    second = PrimaryNode(operation_type='linear')
    final = SecondaryNode(operation_type='kmeans',
                          nodes_from=[first, second])

    pipeline = Pipeline(final)

    return pipeline, Task(TaskTypesEnum.classification)


def pipeline_with_only_data_operations():
    first = PrimaryNode(operation_type='one_hot_encoding')
    second = SecondaryNode(operation_type='scaling', nodes_from=[first])
    final = SecondaryNode(operation_type='ransac_lin_reg', nodes_from=[second])

    pipeline = Pipeline(final)

    return pipeline


def pipeline_with_incorrect_data_flow():
    """ When combining the features in the presented pipeline, a table with 5
    columns will turn into a table with 10 columns """
    first = PrimaryNode(operation_type='scaling')
    second = PrimaryNode(operation_type='scaling')

    final = SecondaryNode(operation_type='ridge', nodes_from=[first, second])
    pipeline = Pipeline(final)
    return pipeline


def ts_pipeline_with_incorrect_data_flow():
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
    pipeline = Pipeline(node_final)

    return pipeline


def pipeline_with_incorrect_parent_number_for_decompose():
    """ Pipeline structure:
           logit
    scaling                        rf
           class_decompose -> rfr
    For class_decompose connection with "logit" model needed
    """

    node_scaling = PrimaryNode('scaling')
    node_logit = SecondaryNode('logit', nodes_from=[node_scaling])
    node_decompose = SecondaryNode('class_decompose', nodes_from=[node_scaling])
    node_rfr = SecondaryNode('rfr', nodes_from=[node_decompose])
    node_rf = SecondaryNode('rf', nodes_from=[node_rfr, node_logit])
    pipeline = Pipeline(node_rf)
    return pipeline


def pipeline_with_incorrect_parents_position_for_decompose():
    """ Pipeline structure:
         scaling
    logit                       rf
         class_decompose -> rfr
    """

    node_first = PrimaryNode('logit')
    node_second = SecondaryNode('scaling', nodes_from=[node_first])
    node_decompose = SecondaryNode('class_decompose', nodes_from=[node_second, node_first])
    node_rfr = SecondaryNode('rfr', nodes_from=[node_decompose])
    node_rf = SecondaryNode('rf', nodes_from=[node_rfr, node_second])
    pipeline = Pipeline(node_rf)
    return pipeline


def correct_decompose_pipeline():
    """
            logit
    scaling                         rf
            class_decompose -> rfr
    """
    node_first = PrimaryNode('scaling')
    node_second = SecondaryNode('logit', nodes_from=[node_first])
    node_decompose = SecondaryNode('class_decompose', nodes_from=[node_second, node_first])
    node_rfr = SecondaryNode('rfr', nodes_from=[node_decompose])
    node_rf = SecondaryNode('rf', nodes_from=[node_rfr, node_second])
    pipeline = Pipeline(node_rf)
    return pipeline


def pipeline_with_correct_data_sources():
    node_first = PrimaryNode('data_source/1')
    node_second = PrimaryNode('data_source/2')
    pipeline = Pipeline(SecondaryNode('linear', [node_first, node_second]))
    return pipeline


def pipeline_with_incorrect_data_sources():
    node_first = PrimaryNode('data_source/1')
    node_second = PrimaryNode('scaling')
    pipeline = Pipeline(SecondaryNode('linear', [node_first, node_second]))
    return pipeline


def pipeline_with_incorrect_resample_node():
    """ Incorrect pipeline
        resample \
                   model
        scaling  /
    """
    resample_node = PrimaryNode(operation_type='resample')
    scaling_node = PrimaryNode(operation_type='scaling')
    pipeline = Pipeline(SecondaryNode(operation_type='logit', nodes_from=[resample_node, scaling_node]))

    return pipeline


def test_pipeline_with_cycle_raise_exception():
    pipeline = pipeline_with_cycle()
    with pytest.raises(Exception) as exc:
        assert has_no_cycle(pipeline)
    assert str(exc.value) == f'{GRAPH_ERROR_PREFIX} Graph has cycles'


def test_pipeline_without_cycles_correct():
    pipeline = valid_pipeline()

    assert has_no_cycle(pipeline)


def test_pipeline_with_isolated_nodes_raise_exception():
    pipeline = pipeline_with_isolated_nodes()
    with pytest.raises(ValueError) as exc:
        assert has_no_isolated_nodes(pipeline)
    assert str(exc.value) == f'{GRAPH_ERROR_PREFIX} Graph has isolated nodes'


def test_multi_root_pipeline_raise_exception():
    pipeline = pipeline_with_multiple_roots()

    with pytest.raises(Exception) as exc:
        assert pipeline.root_node
    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} More than 1 root_nodes in pipeline'


def test_pipeline_with_primary_nodes_correct():
    pipeline = valid_pipeline()
    assert has_primary_nodes(pipeline)


def test_pipeline_without_primary_nodes_raise_exception():
    pipeline = pipeline_with_secondary_nodes_only()
    with pytest.raises(Exception) as exc:
        assert has_primary_nodes(pipeline)
    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Pipeline does not have primary nodes'


def test_pipeline_with_self_cycled_nodes_raise_exception():
    pipeline = pipeline_with_self_cycle()
    with pytest.raises(Exception) as exc:
        assert has_no_self_cycled_nodes(pipeline)
    assert str(exc.value) == f'{GRAPH_ERROR_PREFIX} Graph has self-cycled nodes'


def test_pipeline_validate_correct():
    pipeline = valid_pipeline()
    validate(pipeline)


def test_pipeline_with_isolated_components_raise_exception():
    pipeline = pipeline_with_isolated_components()
    with pytest.raises(Exception) as exc:
        assert has_no_isolated_components(pipeline)
    assert str(exc.value) == f'{GRAPH_ERROR_PREFIX} Graph has isolated components'


def test_pipeline_with_incorrect_task_type_raise_exception():
    pipeline, task = pipeline_with_incorrect_task_type()
    with pytest.raises(Exception) as exc:
        assert has_correct_operation_positions(pipeline, task)
    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Pipeline has incorrect operations positions'


def test_pipeline_without_model_in_root_node():
    incorrect_pipeline = pipeline_with_only_data_operations()

    with pytest.raises(Exception) as exc:
        assert has_final_operation_as_model(incorrect_pipeline)

    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Root operation is not a model'


def test_pipeline_with_incorrect_data_flow():
    incorrect_pipeline = pipeline_with_incorrect_data_flow()

    with pytest.raises(Exception) as exc:
        assert has_no_conflicts_with_data_flow(incorrect_pipeline)

    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Pipeline has incorrect subgraph with identical data operations'


def test_ts_pipeline_with_incorrect_data_flow():
    incorrect_pipeline = ts_pipeline_with_incorrect_data_flow()

    if is_pipeline_contains_ts_operations(incorrect_pipeline):
        with pytest.raises(Exception) as exc:
            assert has_no_data_flow_conflicts_in_ts_pipeline(incorrect_pipeline)

        assert str(exc.value) == \
               f'{PIPELINE_ERROR_PREFIX} Pipeline has incorrect subgraph with wrong parent nodes combination'
    else:
        assert False


def test_only_non_lagged_operations_are_primary():
    """ Incorrect pipeline
    lagged \
             linear -> final forecast
     ridge /
    """
    node_lagged = PrimaryNode('lagged')
    node_ridge = PrimaryNode('ridge')
    node_final = SecondaryNode('linear', nodes_from=[node_lagged, node_ridge])
    incorrect_pipeline = Pipeline(node_final)

    with pytest.raises(Exception) as exc:
        assert only_non_lagged_operations_are_primary(incorrect_pipeline)

    assert str(exc.value) == \
           f'{PIPELINE_ERROR_PREFIX} Pipeline for forecasting has not non_lagged preprocessing in primary nodes'


def test_has_two_parents_for_decompose_operations():
    incorrect_pipeline = pipeline_with_incorrect_parent_number_for_decompose()

    with pytest.raises(Exception) as exc:
        assert has_no_conflicts_in_decompose(incorrect_pipeline)

    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Two parents for decompose node were expected, but 1 were given'


def test_decompose_parents_has_wright_positions():
    incorrect_pipeline = pipeline_with_incorrect_parents_position_for_decompose()

    with pytest.raises(Exception) as exc:
        assert has_no_conflicts_in_decompose(incorrect_pipeline)

    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} For decompose operation Model as first parent is required'


def test_decompose_operation_remove_in_pipeline():
    """
    In the process of evolution, edges and nodes may have been removed from the decompose pipeline.
    Or a decompose node could be replaced with a new one. Such pipelines are incorrect.
    In this test replacement the class_decompose with logit operation is performed
    """
    current_pipeline = correct_decompose_pipeline()
    for node in current_pipeline.nodes:
        if node.operation.operation_type == 'class_decompose':
            # Replace decompose node with simple classification model
            node.operation.operation_type = 'logit'

    with pytest.raises(ValueError) as exc:
        has_no_conflicts_during_multitask(current_pipeline)

    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Current pipeline can not solve multitask problem'


def test_incorrect_node_after_decompose_operation():
    """
    The regression model should be next after class_decompose operation.
    If it doesn't, then the pipelining is incorrect
    """
    current_pipeline = correct_decompose_pipeline()
    for node in current_pipeline.nodes:
        if node.operation.operation_type == 'rfr':
            # Replace regression model with classification one
            node.operation.operation_type = 'lda'

    with pytest.raises(ValueError) as exc:
        has_no_conflicts_after_class_decompose(current_pipeline)

    expected_error = f'{PIPELINE_ERROR_PREFIX} After classification decompose it is required to use regression model'
    assert str(exc.value) == expected_error


def test_data_sources_validation():
    incorrect_pipeline = pipeline_with_incorrect_data_sources()

    with pytest.raises(ValueError) as exc:
        has_correct_data_sources(incorrect_pipeline)

    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Data sources are mixed with other primary nodes'

    correct_pipeline = pipeline_with_correct_data_sources()
    assert has_correct_data_sources(correct_pipeline)


def test_custom_validation():
    incorrect_pipeline = pipeline_with_incorrect_parents_position_for_decompose()

    assert validate(incorrect_pipeline, rules=[has_no_cycle])


def test_pipeline_with_resample_node():
    incorrect_pipeline = pipeline_with_incorrect_resample_node()

    with pytest.raises(ValueError) as exc:
        has_parent_contain_single_resample(incorrect_pipeline)

    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Resample node is not single parent node for child operation'
