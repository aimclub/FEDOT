from fedot.core.pipelines.pipeline_builder import PipelineBuilder


def linear_pipeline():
    pipeline = PipelineBuilder().add_node('knnreg').add_node('ridge').build()
    return pipeline


def simple_branched_pipeline():
    pipeline = PipelineBuilder().add_branch('svr', 'ridge').join_branches('rfr').build()
    return pipeline


def complex_branched_pipeline():
    pipeline = PipelineBuilder().add_branch('svr', 'lasso', 'ridge').grow_branches('rfr', 'rfr', 'dtreg')\
        .grow_branches('ridge', 'svr', 'ridge').join_branches('rfr').build()
    return pipeline


def get_pipelines_for_regression():
    return {'linear': linear_pipeline(), 'branched': simple_branched_pipeline(), 'complex': complex_branched_pipeline()}

