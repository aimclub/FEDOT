from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


def ts_ets_pipeline():
    """
    Return pipeline with the following structure:
    cut -> ets -> final forecast
    Where cut - cut part of dataset and ets - exponential smoothing
    """
    node_cut = PrimaryNode("cut")
    node_ets = SecondaryNode("ets", nodes_from=[node_cut])
    node_ets.custom_params = {"error": "add",
                              "trend": "add",
                              "seasonal": "add",
                              "damped_trend": False,
                              "seasonal_periods": 20}
    return Pipeline(node_ets)


def ts_ets_ridge_pipeline():
    """
    Return pipeline with the following structure:
       cut -  ets \
                   -> ridge -> final forecast
    lagged - ridge /
    Where cut - cut part of dataset, ets - exponential smoothing
   """
    node_cut = PrimaryNode("cut")
    node_cut.custom_params = {"cut_part": 0.5}
    node_ets = SecondaryNode("ets", nodes_from=[node_cut])
    node_ets.custom_params = {"error": "add",
                              "trend": "add",
                              "seasonal": "add",
                              "damped_trend": False,
                              "seasonal_periods": 20}

    node_lagged = PrimaryNode("lagged")
    node_ridge_1 = SecondaryNode("ridge", nodes_from=[node_lagged])

    node_ridge_2 = SecondaryNode("ridge", nodes_from=[node_ridge_1, node_ets])

    return Pipeline(node_ridge_2)


def ts_glm_pipeline():
    """
    Return pipeline with the following structure:
    glm -> final forecast

    Where glm - Generalized linear model
    """
    node_glm = PrimaryNode("glm")
    node_glm.custom_params = {"family": "gaussian"}
    return Pipeline(node_glm)


def ts_glm_ridge_pipeline():
    """
    Return pipeline with the following structure:
               glm \
                   -> ridge -> final forecast
    lagged - ridge /

    Where glm - Generalized linear model
    """
    node_glm = PrimaryNode("glm")

    node_lagged = PrimaryNode("lagged")
    node_ridge_1 = SecondaryNode("ridge", nodes_from=[node_lagged])

    node_ridge_2 = SecondaryNode("ridge", nodes_from=[node_ridge_1, node_glm])

    return Pipeline(node_ridge_2)


def ts_polyfit_pipeline(degree):
    """
    Return pipeline with the following structure:
    polyfit -> final forecast

    Where polyfit - Polynomial interpolation
    """
    node_polyfit = PrimaryNode("polyfit")
    node_polyfit.custom_params = {"degree": degree}
    return Pipeline(node_polyfit)


def ts_polyfit_ridge_pipeline(degree):
    """
    Return pipeline with the following structure:
           polyfit \
                   -> ridge -> final forecast
    lagged - ridge /

    Where polyfit - Polynomial interpolation
    """
    node_polyfit = PrimaryNode("polyfit")
    node_polyfit.custom_params = {"degree": degree}

    node_lagged = PrimaryNode("lagged")
    node_ridge_1 = SecondaryNode("ridge", nodes_from=[node_lagged])

    node_ridge_2 = SecondaryNode("ridge", nodes_from=[node_ridge_1, node_polyfit])

    return Pipeline(node_ridge_2)


def ts_complex_ridge_pipeline():
    """
    Return pipeline with the following structure:
    lagged - ridge \
                    -> ridge -> final forecast
    lagged - ridge /
    """
    node_lagged_1 = PrimaryNode("lagged")
    node_lagged_2 = PrimaryNode("lagged")

    node_ridge_1 = SecondaryNode("ridge", nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode("ridge", nodes_from=[node_lagged_2])

    node_final = SecondaryNode("ridge", nodes_from=[node_ridge_1, node_ridge_2])
    pipeline = Pipeline(node_final)

    return pipeline


def ts_complex_ridge_smoothing_pipeline():
    """
    Pipeline looking like this
    smoothing - lagged - ridge \
                                \
                                 ridge -> final forecast
                                /
                lagged - ridge /

    Where smoothing - rolling mean
    """
    node_smoothing = PrimaryNode("smoothing")
    node_lagged_1 = SecondaryNode("lagged", nodes_from=[node_smoothing])
    node_lagged_2 = PrimaryNode("lagged")

    node_ridge_1 = SecondaryNode("ridge", nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode("ridge", nodes_from=[node_lagged_2])

    node_final = SecondaryNode("ridge", nodes_from=[node_ridge_1, node_ridge_2])
    pipeline = Pipeline(node_final)

    return pipeline


def ts_complex_dtreg_pipeline(first_node="lagged"):
    """
        Return pipeline with the following structure:

        lagged/sparse_lagged - dtreg \
                                        rfr -> final forecast
        lagged/sparse_lagged - dtreg /

        Where dtreg = tree regressor, rfr - random forest regressor
    """
    node_lagged_1 = PrimaryNode(first_node)
    node_lagged_2 = PrimaryNode(first_node)
    node_dtreg_1 = SecondaryNode("dtreg", nodes_from=[node_lagged_1])
    node_dtreg_2 = SecondaryNode("dtreg", nodes_from=[node_lagged_2])
    node_final = SecondaryNode("rfr", nodes_from=[node_dtreg_1, node_dtreg_2])
    pipeline = Pipeline(node_final)
    return pipeline


def ts_multiple_ets_pipeline():
    """
    Return pipeline with the following structure:
      ets
         \
    ets -> lasso -> final forecast
        /
     ets

    Where ets - exponential_smoothing
    """
    node_ets2 = PrimaryNode("ets")
    node_ets = PrimaryNode("ets")
    node_ets3 = PrimaryNode("ets")
    final_lasso = SecondaryNode('lasso', nodes_from=[node_ets, node_ets2, node_ets3])
    pipeline = Pipeline(final_lasso)
    return pipeline


def ts_ar_pipeline():
    """
    Return pipeline with the following structure:
    ar -> final forecast

    Where ar - auto regression
    """
    node_ar = PrimaryNode("ar")
    pipeline = Pipeline(node_ar)
    return pipeline


def clstm_pipeline():
    """
    Return pipeline with the following structure:
    lagged - ridge \
                    -> ridge -> final forecast
             clstm /

    Where clstm - convolutional long short-term memory model
    """
    node_lagged_1 = PrimaryNode("lagged")

    node_ridge_1 = SecondaryNode("ridge", nodes_from=[node_lagged_1])
    node_clstm = PrimaryNode("clstm")
    node_clstm.custom_params = {"window_size": 29,
                                "hidden_size": 50,
                                "learning_rate": 0.004,
                                "cnn1_kernel_size": 5,
                                "cnn1_output_size": 32,
                                "cnn2_kernel_size": 4,
                                "cnn2_output_size": 32,
                                "batch_size": 64,
                                "num_epochs": 3}

    node_final = SecondaryNode("ridge", nodes_from=[node_ridge_1, node_clstm])
    pipeline = Pipeline(node_final)

    return pipeline
