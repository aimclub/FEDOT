from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


def regression_three_depth_manual_pipeline():
    """
    Returns pipeline with the following structure:

          xgbreg -> dtreg  \
                             knnreg - > final prediction
             knnreg -> rfr /

    Where xgboost - xg boosting algorithm, dtreg - tree regression, knn - K nearest neighbors regression,
   """
    xgb_primary = PrimaryNode('xgbreg')
    knn_primary = PrimaryNode('knnreg')

    dtreg_secondary = SecondaryNode('dtreg', nodes_from=[xgb_primary])
    rfr_secondary = SecondaryNode('rfr', nodes_from=[knn_primary])

    knnreg_root = SecondaryNode('knnreg', nodes_from=[dtreg_secondary, rfr_secondary])

    pipeline = Pipeline(knnreg_root)

    return pipeline


def regression_ransac_pipeline():
    """
    Returns pipeline with the following structure:

          scaling -> ransac_lin_reg -> lasso -> final prediction

    Where ransac_lin_reg - ransac algorithm
   """
    node_scaling = PrimaryNode('scaling')
    node_ransac = SecondaryNode('ransac_lin_reg', nodes_from=[node_scaling])
    node_ridge = SecondaryNode('lasso', nodes_from=[node_ransac])
    pipeline = Pipeline(node_ridge)
    return pipeline
