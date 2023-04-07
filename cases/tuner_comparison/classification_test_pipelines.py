from fedot.core.pipelines.pipeline_builder import PipelineBuilder


def linear_pipeline():
    pipeline = PipelineBuilder().add_node('knn').add_node('rf').build()
    return pipeline


def simple_branched_pipeline():
    pipeline = PipelineBuilder().add_branch('dt', 'knn').join_branches('rf').build()
    return pipeline


def complex_branched_pipeline():
    pipeline = PipelineBuilder().add_branch('knn', 'dt', 'rf').grow_branches('dt', 'knn', 'rf')\
        .grow_branches('rf', 'knn', 'dt').add_skip_connection_edge(2, 1, 1, 0).join_branches('rf').build()
    return pipeline


def get_pipelines_for_classification():
    return {'linear': linear_pipeline(), 'branched': simple_branched_pipeline(), 'complex': complex_branched_pipeline()}
