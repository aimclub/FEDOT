from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.operations.atomized_model import AtomizedModel


def sample_pipeline():
    return Pipeline(SecondaryNode(operation_type='logit',
                                  nodes_from=[PrimaryNode(operation_type='xgboost'),
                                              PrimaryNode(operation_type='scaling')]))


def generate_pipeline() -> Pipeline:
    node_scaling = PrimaryNode('scaling')
    node_first = SecondaryNode('kmeans', nodes_from=[node_scaling])
    node_second = SecondaryNode('kmeans', nodes_from=[node_scaling])
    node_root = SecondaryNode('logit', nodes_from=[node_first, node_second])
    pipeline = Pipeline(node_root)
    return pipeline


def pipeline_simple() -> Pipeline:
    node_scaling = PrimaryNode('scaling')
    node_svc = SecondaryNode('svc', nodes_from=[node_scaling])
    node_lda = SecondaryNode('lda', nodes_from=[node_scaling])
    node_final = SecondaryNode('rf', nodes_from=[node_svc, node_lda])

    pipeline = Pipeline(node_final)

    return pipeline


def pipeline_with_pca() -> Pipeline:
    node_scaling = PrimaryNode('scaling')
    node_pca = SecondaryNode('pca', nodes_from=[node_scaling])
    node_lda = SecondaryNode('lda', nodes_from=[node_scaling])
    node_final = SecondaryNode('rf', nodes_from=[node_pca, node_lda])

    pipeline = Pipeline(node_final)

    return pipeline


def valid_pipeline():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit',
                           nodes_from=[first])
    third = SecondaryNode(operation_type='logit',
                          nodes_from=[second])
    last = SecondaryNode(operation_type='logit',
                         nodes_from=[third])

    pipeline = Pipeline(last)

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


def pipeline_with_incorrect_parent_amount_for_decompose():
    """ Pipeline structure:
           logit
    scaling                        xgboost
           class_decompose -> rfr
    For class_decompose connection with "logit" model needed
    """

    node_scaling = PrimaryNode('scaling')
    node_logit = SecondaryNode('logit', nodes_from=[node_scaling])
    node_decompose = SecondaryNode('class_decompose', nodes_from=[node_scaling])
    node_rfr = SecondaryNode('rfr', nodes_from=[node_decompose])
    node_xgboost = SecondaryNode('xgboost', nodes_from=[node_rfr, node_logit])
    pipeline = Pipeline(node_xgboost)
    return pipeline


def pipeline_with_incorrect_parents_position_for_decompose():
    """ Pipeline structure:
         scaling
    logit                       xgboost
         class_decompose -> rfr
    """

    node_first = PrimaryNode('logit')
    node_second = SecondaryNode('scaling', nodes_from=[node_first])
    node_decompose = SecondaryNode('class_decompose', nodes_from=[node_second, node_first])
    node_rfr = SecondaryNode('rfr', nodes_from=[node_decompose])
    node_xgboost = SecondaryNode('xgboost', nodes_from=[node_rfr, node_second])
    pipeline = Pipeline(node_xgboost)
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


def get_simple_regr_pipeline():
    final = PrimaryNode(operation_type='xgbreg')
    pipeline = Pipeline(final)

    return pipeline


def get_complex_regr_pipeline():
    node_scaling = PrimaryNode(operation_type='scaling')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_scaling])
    node_linear = SecondaryNode('linear', nodes_from=[node_scaling])
    final = SecondaryNode('xgbreg', nodes_from=[node_ridge, node_linear])
    pipeline = Pipeline(final)

    return pipeline


def get_simple_class_pipeline():
    final = PrimaryNode(operation_type='logit')
    pipeline = Pipeline(final)

    return pipeline


def get_complex_class_pipeline():
    first = PrimaryNode(operation_type='xgboost')
    second = PrimaryNode(operation_type='pca')
    final = SecondaryNode(operation_type='logit',
                          nodes_from=[first, second])

    pipeline = Pipeline(final)

    return pipeline


def get_simple_short_lagged_pipeline():
    # Create simple pipeline for forecasting
    node_lagged = PrimaryNode('lagged')
    # Use 4 elements in time series as predictors
    node_lagged.custom_params = {'window_size': 4}
    node_final = SecondaryNode('linear', nodes_from=[node_lagged])
    pipeline = Pipeline(node_final)

    return pipeline


def pipeline_first():
    #    XG
    #  |     \
    # XG      KNN
    # |  \    |  \
    # LR LDA LR  LDA
    pipeline = Pipeline()

    root_of_tree, root_child_first, root_child_second = \
        [SecondaryNode(model) for model in ('xgboost', 'xgboost', 'knn')]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in ('logit', 'lda'):
            new_node = PrimaryNode(requirement_model)
            root_node_child.nodes_from.append(new_node)
            pipeline.add_node(new_node)
        pipeline.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    pipeline.add_node(root_of_tree)
    return pipeline


def pipeline_second():
    #    XG
    #  |     \
    # DT      KNN
    # |  \    |  \
    # KNN KNN LR  LDA
    pipeline = pipeline_first()
    new_node = SecondaryNode('dt')
    for model_type in ('knn', 'knn'):
        new_node.nodes_from.append(PrimaryNode(model_type))
    pipeline.update_subtree(pipeline.root_node.nodes_from[0], new_node)
    return pipeline


def pipeline_third():
    #    QDA
    #  |     \
    # RF     RF
    pipeline = Pipeline()
    new_node = SecondaryNode('qda')
    for model_type in ('rf', 'rf'):
        new_node.nodes_from.append(PrimaryNode(model_type))
    pipeline.add_node(new_node)
    [pipeline.add_node(node_from) for node_from in new_node.nodes_from]
    return pipeline


def pipeline_fourth():
    #          XG
    #      |         \
    #     XG          KNN
    #   |    \        |  \
    # QDA     KNN     LR  LDA
    # |  \    |    \
    # RF  RF  KNN KNN
    pipeline = pipeline_first()
    new_node = SecondaryNode('qda')
    for model_type in ('rf', 'rf'):
        new_node.nodes_from.append(PrimaryNode(model_type))
    pipeline.update_subtree(pipeline.root_node.nodes_from[0].nodes_from[1], new_node)
    new_node = SecondaryNode('knn')
    for model_type in ('knn', 'knn'):
        new_node.nodes_from.append(PrimaryNode(model_type))
    pipeline.update_subtree(pipeline.root_node.nodes_from[0].nodes_from[0], new_node)
    return pipeline


def pipeline_fifth():
    #    KNN
    #  |     \
    # XG      KNN
    # |  \    |  \
    # LR LDA KNN  KNN
    pipeline = pipeline_first()
    new_node = SecondaryNode('knn')
    pipeline.update_node(pipeline.root_node, new_node)
    new_node = PrimaryNode('knn')
    pipeline.update_node(pipeline.root_node.nodes_from[1].nodes_from[0], new_node)
    pipeline.update_node(pipeline.root_node.nodes_from[1].nodes_from[1], new_node)

    return pipeline


def generate_pipeline_with_decomposition(primary_operation, secondary_operation):
    """ The function generates a pipeline in which there is an operation of
    decomposing the target variable into residuals
                     secondary_operation
    primary_operation                       xgboost
                     class_decompose -> rfr

    :param primary_operation: name of operation to place in primary node
    :param secondary_operation: name of operation to place in secondary node
    """

    node_first = PrimaryNode(primary_operation)
    node_second = SecondaryNode(secondary_operation, nodes_from=[node_first])
    node_decompose = SecondaryNode('class_decompose', nodes_from=[node_second, node_first])
    node_rfr = SecondaryNode('rfr', nodes_from=[node_decompose])
    node_xgboost = SecondaryNode('xgboost', nodes_from=[node_rfr, node_second])
    full_pipeline = Pipeline(node_xgboost)
    return full_pipeline


def generate_pipeline_with_filtering():
    """ Return 5-level pipeline with decompose and filtering operations
           logit
    scaling                                 xgboost
           class_decompose -> RANSAC -> rfr
    """

    node_scaling = PrimaryNode('scaling')
    node_logit = SecondaryNode('logit', nodes_from=[node_scaling])
    node_decompose = SecondaryNode('class_decompose', nodes_from=[node_logit, node_scaling])
    node_ransac = SecondaryNode('ransac_lin_reg', nodes_from=[node_decompose])
    node_rfr = SecondaryNode('rfr', nodes_from=[node_ransac])
    node_xgboost = SecondaryNode('xgboost', nodes_from=[node_rfr, node_logit])
    full_pipeline = Pipeline(node_xgboost)
    return full_pipeline


def generate_cascade_decompose_pipeline():
    """ The function of generating a multi-stage model with many connections
    and solving many problems (regression and classification)
    """

    node_scaling = PrimaryNode('scaling')
    node_second = SecondaryNode('logit', nodes_from=[node_scaling])
    node_decompose = SecondaryNode('class_decompose', nodes_from=[node_second, node_scaling])
    node_rfr = SecondaryNode('rfr', nodes_from=[node_decompose])
    node_xgboost = SecondaryNode('xgboost', nodes_from=[node_rfr, node_second])
    node_decompose_new = SecondaryNode('class_decompose', nodes_from=[node_xgboost, node_scaling])
    node_rfr_2 = SecondaryNode('rfr', nodes_from=[node_decompose_new])
    node_final = SecondaryNode('logit', nodes_from=[node_rfr_2, node_xgboost])
    pipeline = Pipeline(node_final)
    return pipeline


def get_refinement_pipeline(lagged):
    """ Create 4-level pipeline with decompose operation """

    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': lagged}
    node_lasso = SecondaryNode('lasso', nodes_from=[node_lagged])
    node_decompose = SecondaryNode('decompose', nodes_from=[node_lagged, node_lasso])
    node_dtreg = SecondaryNode('dtreg', nodes_from=[node_decompose])
    node_dtreg.custom_params = {'max_depth': 3}

    # Pipelines with different outputs
    pipeline_with_decompose_finish = Pipeline(node_dtreg)
    pipeline_with_main_finish = Pipeline(node_lasso)

    # Combining branches with different targets (T and T_decomposed)
    final_node = SecondaryNode('ridge', nodes_from=[node_lasso, node_dtreg])

    pipeline = Pipeline(final_node)
    return pipeline_with_main_finish, pipeline_with_decompose_finish, pipeline


def get_non_refinement_pipeline(lagged):
    """ Create 4-level pipeline without decompose operation """

    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': lagged}
    node_lasso = SecondaryNode('lasso', nodes_from=[node_lagged])
    node_dtreg = SecondaryNode('dtreg', nodes_from=[node_lagged])
    node_dtreg.custom_params = {'max_depth': 3}
    final_node = SecondaryNode('ridge', nodes_from=[node_lasso, node_dtreg])

    pipeline = Pipeline(final_node)
    return pipeline


def get_knn_reg_pipeline(k_neighbors):
    """ Function return pipeline with K-nn regression model in it """
    node_scaling = PrimaryNode('scaling')
    node_final = SecondaryNode('knnreg', nodes_from=[node_scaling])
    node_final.custom_params = {'n_neighbors': k_neighbors}
    pipeline = Pipeline(node_final)
    return pipeline


def get_knn_class_pipeline(k_neighbors):
    """ Function return pipeline with K-nn classification model in it """
    node_scaling = PrimaryNode('scaling')
    node_final = SecondaryNode('knn', nodes_from=[node_scaling])
    node_final.custom_params = {'n_neighbors': k_neighbors}
    pipeline = Pipeline(node_final)
    return pipeline


def generate_pipeline() -> Pipeline:
    pipeline = Pipeline(PrimaryNode('logit'))
    return pipeline


def create_pipeline() -> Pipeline:
    node_logit = PrimaryNode('logit')

    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'n_components': 1}

    node_xgboost = PrimaryNode('xgboost')

    node_knn = PrimaryNode('knn')
    node_knn.custom_params = {'n_neighbors': 9}

    node_knn_second = SecondaryNode('knn')
    node_knn_second.custom_params = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_lda, node_knn]

    node_logit_second = SecondaryNode('logit')
    node_logit_second.nodes_from = [node_xgboost, node_lda]

    node_lda_second = SecondaryNode('lda')
    node_lda_second.custom_params = {'n_components': 1}
    node_lda_second.nodes_from = [node_logit_second, node_knn_second, node_logit]

    node_xgboost_second = SecondaryNode('xgboost')
    node_xgboost_second.nodes_from = [node_logit, node_logit_second, node_knn]

    node_knn_third = SecondaryNode('knn')
    node_knn_third.custom_params = {'n_neighbors': 8}
    node_knn_third.nodes_from = [node_lda_second, node_xgboost_second]

    pipeline = Pipeline(node_knn_third)

    return pipeline


def create_fitted_pipeline() -> Pipeline:
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    pipeline = create_pipeline()
    pipeline.fit(train_data)

    return pipeline


def create_classification_pipeline_with_preprocessing():
    node_scaling = PrimaryNode('scaling')
    node_rfe = PrimaryNode('rfe_lin_class')

    xgb_node = SecondaryNode('xgboost', nodes_from=[node_scaling])
    logit_node = SecondaryNode('logit', nodes_from=[node_rfe])

    knn_root = SecondaryNode('knn', nodes_from=[xgb_node, logit_node])

    pipeline = Pipeline(knn_root)

    return pipeline


def create_four_depth_pipeline():
    knn_node = PrimaryNode('knn')
    lda_node = PrimaryNode('lda')
    xgb_node = PrimaryNode('xgboost')
    logit_node = PrimaryNode('logit')

    logit_node_second = SecondaryNode('logit', nodes_from=[knn_node, lda_node])
    xgb_node_second = SecondaryNode('xgboost', nodes_from=[logit_node])

    qda_node_third = SecondaryNode('qda', nodes_from=[xgb_node_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[logit_node_second, xgb_node])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    pipeline = Pipeline(knn_root)

    return pipeline


def get_multiscale_pipeline():
    # First branch
    node_lagged_1 = PrimaryNode('lagged')
    node_lagged_1.custom_params = {'window_size': 20}
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])

    # Second branch, which will try to make prediction based on smoothed ts
    node_filtering = PrimaryNode('gaussian_filter')
    node_filtering.custom_params = {'sigma': 3}
    node_lagged_2 = SecondaryNode('lagged', nodes_from=[node_filtering])
    node_lagged_2.custom_params = {'window_size': 100}
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    node_final = SecondaryNode('linear', nodes_from=[node_ridge_1, node_ridge_2])

    pipeline = Pipeline(node_final)

    return pipeline


def get_simple_ts_pipeline(model_root: str = 'ridge', window_size: int = 20):
    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': window_size}
    node_root = SecondaryNode(model_root, nodes_from=[node_lagged])

    pipeline = Pipeline(node_root)

    return pipeline


def get_statsmodels_pipeline():
    node_ar = PrimaryNode('ar')
    node_ar.custom_params = {'lag_1': 20, 'lag_2': 100}
    pipeline = Pipeline(node_ar)
    return pipeline


def create_pipeline() -> Pipeline:
    pipeline = Pipeline()
    node_logit = PrimaryNode('logit')

    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'n_components': 1}

    node_xgboost = SecondaryNode('xgboost')
    node_xgboost.custom_params = {'n_components': 1}
    node_xgboost.nodes_from = [node_logit, node_lda]

    pipeline.add_node(node_xgboost)

    return pipeline


def create_atomized_model() -> AtomizedModel:
    """
    Example, how to create Atomized operation.
    """
    pipeline = create_pipeline()
    atomized_model = AtomizedModel(pipeline)

    return atomized_model


def create_atomized_model_with_several_atomized_models() -> AtomizedModel:
    pipeline = Pipeline()
    node_atomized_model_primary = PrimaryNode(operation_type=create_atomized_model())
    node_atomized_model_secondary = SecondaryNode(operation_type=create_atomized_model())
    node_atomized_model_secondary_second = SecondaryNode(operation_type=create_atomized_model())
    node_atomized_model_secondary_third = SecondaryNode(operation_type=create_atomized_model())

    node_atomized_model_secondary.nodes_from = [node_atomized_model_primary]
    node_atomized_model_secondary_second.nodes_from = [node_atomized_model_primary]
    node_atomized_model_secondary_third.nodes_from = [node_atomized_model_secondary,
                                                      node_atomized_model_secondary_second]

    pipeline.add_node(node_atomized_model_secondary_third)
    atomized_model = AtomizedModel(pipeline)

    return atomized_model


def create_pipeline_with_several_nested_atomized_model() -> Pipeline:
    pipeline = Pipeline()
    atomized_op = create_atomized_model_with_several_atomized_models()
    node_atomized_model = PrimaryNode(operation_type=atomized_op)

    node_atomized_model_secondary = SecondaryNode(operation_type=create_atomized_model())
    node_atomized_model_secondary.nodes_from = [node_atomized_model]

    node_knn = SecondaryNode('knn')
    node_knn.custom_params = {'n_neighbors': 9}
    node_knn.nodes_from = [node_atomized_model]

    node_knn_second = SecondaryNode('knn')
    node_knn_second.custom_params = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_atomized_model, node_atomized_model_secondary, node_knn]

    node_atomized_model_secondary_second = \
        SecondaryNode(operation_type=create_atomized_model_with_several_atomized_models())

    node_atomized_model_secondary_second.nodes_from = [node_knn_second]

    pipeline.add_node(node_atomized_model_secondary_second)

    return pipeline


def get_ts_pipeline(window_size):
    """ Function return pipeline with lagged transformation in it """
    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': window_size}

    node_final = SecondaryNode('ridge', nodes_from=[node_lagged])
    pipeline = Pipeline(node_final)
    return pipeline


def get_ransac_pipeline():
    """ Function return pipeline with lagged transformation in it """
    node_ransac = PrimaryNode('ransac_lin_reg')
    node_final = SecondaryNode('linear', nodes_from=[node_ransac])
    pipeline = Pipeline(node_final)
    return pipeline


def generate_straight_pipeline():
    """ Simple linear pipeline """
    node_scaling = PrimaryNode('scaling')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_scaling])
    node_linear = SecondaryNode('linear', nodes_from=[node_ridge])
    pipeline = Pipeline(node_linear)
    return pipeline


def get_nodes():
    first_node = PrimaryNode('knn')
    second_node = PrimaryNode('knn')
    third_node = SecondaryNode('lda', nodes_from=[first_node, second_node])
    root = SecondaryNode('logit', nodes_from=[third_node])

    return [root, third_node, first_node, second_node]


def get_pipeline():
    third_level_one = PrimaryNode('lda')

    second_level_one = SecondaryNode('qda', nodes_from=[third_level_one])
    second_level_two = PrimaryNode('qda')

    first_level_one = SecondaryNode('knn', nodes_from=[second_level_one, second_level_two])

    root = SecondaryNode(operation_type='logit', nodes_from=[first_level_one])
    pipeline = Pipeline(root)

    return pipeline


def default_valid_pipeline():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit', nodes_from=[first])
    third = SecondaryNode(operation_type='logit', nodes_from=[first])
    final = SecondaryNode(operation_type='logit', nodes_from=[second, third])

    pipeline = Pipeline(final)

    return pipeline


def baseline_pipeline():
    pipeline = Pipeline()
    last_node = SecondaryNode(operation_type='xgboost',
                              nodes_from=[])
    for requirement_model in ['knn', 'logit']:
        new_node = PrimaryNode(requirement_model)
        pipeline.add_node(new_node)
        last_node.nodes_from.append(new_node)
    pipeline.add_node(last_node)

    return pipeline
