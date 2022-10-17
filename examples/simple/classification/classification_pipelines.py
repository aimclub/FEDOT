from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


def cnn_composite_pipeline(composite_flag: bool = True) -> Pipeline:
    """
    Returns pipeline with the following structure:

    .. image:: img_classification_pipelines/cnn_composite_pipeline.png
      :width: 55%

    Where cnn - convolutional neural network, rf - random forest

    :param composite_flag:  add additional random forest estimator
    """
    node_first = PrimaryNode('cnn')
    node_first.parameters = {'architecture': 'deep',
                             'epochs': 15,
                             'batch_size': 128}
    node_second = PrimaryNode('cnn')
    node_second.parameters = {'architecture_type': 'simplified',
                              'epochs': 10,
                              'batch_size': 128}
    node_final = SecondaryNode('rf', nodes_from=[node_first, node_second])

    if not composite_flag:
        node_final = SecondaryNode('rf', nodes_from=[node_first])

    pipeline = Pipeline(node_final)
    return pipeline


def classification_pipeline_with_balancing(custom_params=None):
    """
    Returns pipeline with the following structure:

    .. image:: img_classification_pipelines/class_with_balancing.png
      :width: 55%

    Where resample - algorithm for balancing dataset, logit - logistic_regression

    :param custom_params:  custom parameters for resample node
    """
    node_resample = PrimaryNode(operation_type='resample')

    if custom_params is not None:
        node_resample.parameters = custom_params

    graph = SecondaryNode(operation_type='logit', nodes_from=[node_resample])

    return Pipeline(graph)


def classification_pipeline_without_balancing():
    """
    Returns: pipeline with the following structure:

    .. image:: img_classification_pipelines/class_without_balancing.png
      :width: 55%

    Where logit - logistic_regression
    """
    node = PrimaryNode(operation_type='logit')

    return Pipeline(node)


def classification_complex_pipeline():
    """
    Returns pipeline with the following structure:

    .. image:: img_classification_pipelines/complex_pipeline.png
      :width: 55%

    """
    first = PrimaryNode(operation_type='rf')
    second = PrimaryNode(operation_type='knn')
    final = SecondaryNode(operation_type='logit',
                          nodes_from=[first, second])

    pipeline = Pipeline(final)

    return pipeline


def classification_random_forest_pipeline():
    """
    Returns pipeline with the following structure:

    .. image:: img_classification_pipelines/random_forest.png
      :width: 55%

    """
    node_scaling = PrimaryNode('scaling')
    node_final = SecondaryNode('rf', nodes_from=[node_scaling])
    return Pipeline(node_final)


def classification_isolation_forest_pipeline():
    """
    Returns pipeline with the following structure:

    .. image:: img_classification_pipelines/isolation_forest.png
      :width: 55%

    """
    node_first = PrimaryNode('scaling')
    node_second = SecondaryNode('isolation_forest_class', nodes_from=[node_first])
    node_final = SecondaryNode('rf', nodes_from=[node_second])
    return Pipeline(node_final)


def classification_svc_complex_pipeline():
    """
    Returns pipeline with the following structure:

    .. image:: img_classification_pipelines/svc_complex_pipeline.png
      :width: 55%

    Where svc - support vector classifier, logit - logistic regression, knn - K nearest neighbors classifier,
    rf - random forest classifier

    """

    svc_primary_node = PrimaryNode('svc')
    svc_primary_node.parameters = dict(probability=True)
    logit_secondary_node = SecondaryNode('logit', nodes_from=[svc_primary_node])

    svc_node_with_custom_params = PrimaryNode('svc')
    svc_node_with_custom_params.parameters = dict(kernel='rbf', C=10,
                                                     gamma=1, cache_size=2000,
                                                     probability=True)
    logit_secondary_node_2 = SecondaryNode('logit', nodes_from=[svc_node_with_custom_params])

    knn_primary_node = PrimaryNode('knn')
    knn_secondary_node = SecondaryNode('knn', nodes_from=[knn_primary_node, logit_secondary_node])

    rf_node = SecondaryNode('rf', nodes_from=[logit_secondary_node_2, knn_secondary_node])

    preset_pipeline = Pipeline(rf_node)

    return preset_pipeline


def classification_three_depth_manual_pipeline():
    """
    Returns pipeline with the following structure:

    .. image:: img_classification_pipelines/manual_three_depth_pipeline.png
      :width: 55%

    Where rf - xg boost classifier, logit - logistic regression, knn - K nearest neighbors classifier,
    qda - discriminant analysis
   """
    logit_node_primary = PrimaryNode('logit')
    xgb_node_primary = PrimaryNode('rf')
    xgb_node_primary_second = PrimaryNode('rf')

    qda_node_third = SecondaryNode('qda', nodes_from=[xgb_node_primary_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[logit_node_primary, xgb_node_primary])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    pipeline = Pipeline(knn_root)

    return pipeline


def classification_rf_complex_pipeline():
    """
    Returns pipeline with the following structure:

    .. image:: img_classification_pipelines/complex_rf_pipeline.png
      :width: 55%

    Where lda - discriminant analysis, logit - logistic regression, rf - random forest classifier,
    knn - K nearest neighbors classifier
    """
    pipeline = Pipeline()

    root_of_tree, root_child_first, root_child_second = \
        [SecondaryNode(model) for model in ('rf', 'rf', 'knn')]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in ('logit', 'lda'):
            new_node = PrimaryNode(requirement_model)
            root_node_child.nodes_from.append(new_node)
            pipeline.add_node(new_node)
        pipeline.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    pipeline.add_node(root_of_tree)
    return pipeline
