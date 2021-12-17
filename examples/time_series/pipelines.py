from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


def ets_pipeline():
    node_cut = PrimaryNode('cut')
    node_ets = SecondaryNode('ets', nodes_from=[node_cut])
    node_ets.custom_params = {"error": "add",
                              "trend": "add",
                              'seasonal': "add",
                              "damped_trend": False,
                              "seasonal_periods": 20}
    return Pipeline(node_ets)


def ets_ridge_pipeline():
    node_cut = PrimaryNode('cut')
    node_cut.custom_params = {"cut_part": 0.5}
    node_ets = SecondaryNode('ets', nodes_from=[node_cut])
    node_ets.custom_params = {"error": "add",
                              "trend": "add",
                              'seasonal': "add",
                              "damped_trend": False,
                              "seasonal_periods": 20}
    node_lagged = PrimaryNode('lagged')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    node_ridge1 = SecondaryNode('ridge', nodes_from=[node_ridge, node_ets])

    return Pipeline(node_ridge1)


def glm_pipeline():
    node_glm = PrimaryNode('glm')
    node_glm.custom_params = {"family": "gaussian"}
    return Pipeline(node_glm)


def glm_ridge_pipeline():
    node_glm = PrimaryNode('glm')
    node_glm.custom_params = {"family": "poisson", "link": "log"}
    node_lagged = PrimaryNode('lagged')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    node_ridge1 = SecondaryNode('ridge', nodes_from=[node_ridge, node_glm])

    return Pipeline(node_ridge1)


def polyfit_pipeline(degree):
    node_polyfit = PrimaryNode('polyfit')
    node_polyfit.custom_params = {"degree": degree}
    return Pipeline(node_polyfit)


def polyfit_ridge_pipeline(degree):
    node_polyfit = PrimaryNode('polyfit')
    node_polyfit.custom_params = {"degree": degree}
    node_lagged = PrimaryNode('lagged')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    node_ridge1 = SecondaryNode('ridge', nodes_from=[node_ridge, node_polyfit])

    return Pipeline(node_ridge1)


def complex_rigde_pipeline():
    """
    Return pipeline with the following structure:
    lagged - ridge \
                    -> ridge
    lagged - ridge /
    """

    # First level
    node_lagged_1 = PrimaryNode('lagged')
    node_lagged_2 = PrimaryNode('lagged')

    # Second level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    # Third level - root node
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
    pipeline = Pipeline(node_final)

    return pipeline


def complex_ridge_smoothing_pipeline():
    """
    Pipeline looking like this
    smoothing - lagged - ridge \
                                \
                                 ridge -> final forecast
                                /
                lagged - ridge /
    """

    # First level
    node_smoothing = PrimaryNode('smoothing')

    # Second level
    node_lagged_1 = SecondaryNode('lagged', nodes_from=[node_smoothing])
    node_lagged_2 = PrimaryNode('lagged')

    # Third level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    # Fourth level - root node
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
    pipeline = Pipeline(node_final)

    return pipeline


def complex_dtreg_pipeline(first_node='lagged'):
    """
        Return pipeline with the following structure:

        lagged/sparse_lagged - dtreg \
                                        rfr
        lagged/sparse_lagged - dtreg /
    """

    node_lagged_1 = PrimaryNode(first_node)
    node_lagged_2 = PrimaryNode(first_node)
    node_dtreg_1 = SecondaryNode('dtreg', nodes_from=[node_lagged_1])
    node_dtreg_2 = SecondaryNode('dtreg', nodes_from=[node_lagged_2])
    node_final = SecondaryNode('rfr', nodes_from=[node_dtreg_1, node_dtreg_2])
    pipeline = Pipeline(node_final)
    return pipeline


def ar_pipeline():
    """
    Function return graph with AR model
    """
    node_ar = PrimaryNode('ar')
    pipeline = Pipeline(node_ar)
    return pipeline


def get_source_pipeline_clstm():
    """
    Return pipeline with the following structure:
    lagged - ridge \
                    -> ridge
    clstm - - - - /
    """

    # First level
    node_lagged_1 = PrimaryNode('lagged')

    # Second level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_clstm = PrimaryNode('clstm')
    node_clstm.custom_params = {
        'window_size': 29,
        'hidden_size': 50,
        'learning_rate': 0.004,
        'cnn1_kernel_size': 5,
        'cnn1_output_size': 32,
        'cnn2_kernel_size': 4,
        'cnn2_output_size': 32,
        'batch_size': 64,
        'num_epochs': 3
    }
    # Third level - root node
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_clstm])
    pipeline = Pipeline(node_final)

    return pipeline


def clstm_pipeline():
    """
    Return pipeline with the following structure:
    lagged - ridge \
                    -> ridge
    clstm - - - - /
    """

    # First level
    node_lagged_1 = PrimaryNode('lagged')

    # Second level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_clstm = PrimaryNode('clstm')
    node_clstm.custom_params = {
        'window_size': 29,
        'hidden_size': 50,
        'learning_rate': 0.004,
        'cnn1_kernel_size': 5,
        'cnn1_output_size': 32,
        'cnn2_kernel_size': 4,
        'cnn2_output_size': 32,
        'batch_size': 64,
        'num_epochs': 3
    }
    # Third level - root node
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_clstm])
    pipeline = Pipeline(node_final)

    return pipeline