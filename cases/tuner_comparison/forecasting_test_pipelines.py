from fedot.core.pipelines.pipeline_builder import PipelineBuilder


def linear_pipeline():
    pipeline = PipelineBuilder().add_node('cut').add_node('ets').build()
    return pipeline


def simple_branched_pipeline():
    pipeline = PipelineBuilder().add_branch('lagged', 'lagged').grow_branches('svr', 'ridge').join_branches('rfr')\
        .build()
    return pipeline


def complex_branched_pipeline():
    pipeline = PipelineBuilder().add_branch('smoothing', 'lagged', 'lagged').grow_branches('lagged', 'dtreg', 'ridge')\
        .grow_branches('ridge', 'svr', 'ridge').join_branches('rfr').build()
    return pipeline


def get_pipelines_for_forecasting():
    return {'linear': linear_pipeline(), 'branched': simple_branched_pipeline(), 'complex': complex_branched_pipeline()}

