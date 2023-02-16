from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline


def regression_three_depth_manual_pipeline():
    """
    Returns pipeline with the following structure:

    .. image:: img_regression_pipelines/three_depth_manual.png
      :width: 55%

    Where rf - random forest, dtreg - tree regression, knn - K nearest neighbors regression,
   """
    rfr_primary = PipelineNode('rfr')
    knn_primary = PipelineNode('knnreg')

    dtreg_secondary = PipelineNode('dtreg', nodes_from=[rfr_primary])
    rfr_secondary = PipelineNode('rfr', nodes_from=[knn_primary])

    knnreg_root = PipelineNode('knnreg', nodes_from=[dtreg_secondary, rfr_secondary])

    pipeline = Pipeline(knnreg_root)

    return pipeline


def regression_ransac_pipeline():
    """
    Returns pipeline with the following structure:

    .. image:: img_regression_pipelines/ransac.png
      :width: 55%

    Where ransac_lin_reg - ransac algorithm
   """
    node_scaling = PipelineNode('scaling')
    node_ransac = PipelineNode('ransac_lin_reg', nodes_from=[node_scaling])
    node_ridge = PipelineNode('lasso', nodes_from=[node_ransac])
    pipeline = Pipeline(node_ridge)
    return pipeline
