from fedot.core.log import default_log
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder


def generate_pipeline() -> Pipeline:
    node_scaling = PrimaryNode('scaling')
    node_first = SecondaryNode('kmeans', nodes_from=[node_scaling])
    node_second = SecondaryNode('rf', nodes_from=[node_scaling])
    node_third = SecondaryNode('linear', nodes_from=[node_scaling])
    node_root = SecondaryNode('logit', nodes_from=[node_first, node_second, node_third, node_scaling])

    return Pipeline(node_root)


def show_default(pipeline: Pipeline):
    """ Show with default properties via networkx. """
    pipeline.show()


def show_customized(pipeline: Pipeline):
    """ Show with adjusted sizes and green nodes. """
    pipeline.show(node_color='green', edge_curvature_scale=1.2, node_size_scale=2.0, font_size_scale=1.4)


def show_custom_colors(pipeline: Pipeline):
    """ Show with colors defined by label-color dictionary. """
    pipeline.show(node_color={'scaling': 'tab:olive', 'rf': '#FF7F50', 'linear': (0, 1, 1), None: 'black'},
                  edge_curvature_scale=1.2, node_size_scale=2.0, font_size_scale=1.4)


def show_complex_colors(pipeline: Pipeline):
    """ Show with colors defined by function. """
    def nodes_color(labels):
        if 'xgboost' in labels:
            return {'xgboost': 'tab:orange', None: 'black'}
        else:
            return {'rf': 'tab:green', None: 'black'}

    pipeline.show(node_color=nodes_color)


def show_pyvis(pipeline: Pipeline):
    """ Show with pyvis. """
    pipeline.show(engine='pyvis')


def show_pyvis_custom_colors(pipeline: Pipeline):
    """ Show with pyvis with custom colors. """
    pipeline.show(engine='pyvis',
                  node_color={'scaling': 'tab:olive', 'rf': '#FF7F50', 'linear': (0, 1, 1), None: 'black'})


def show_graphviz(pipeline: Pipeline):
    """ Show with graphviz (requires Graphviz and pygraphviz). """
    pipeline.show(engine='graphviz')


def show_graphviz_custom_colors(pipeline: Pipeline):
    """ Show with graphviz with custom colors (requires Graphviz and pygraphviz). """
    pipeline.show(engine='graphviz',
                  node_color={'scaling': 'tab:olive', 'rf': '#FF7F50', 'linear': (0, 1, 1), None: 'black'})


def main():
    pipeline = generate_pipeline()
    show_default(pipeline)
    show_customized(pipeline)
    show_custom_colors(pipeline)
    show_complex_colors(pipeline)
    show_complex_colors(PipelineBuilder(*pipeline.nodes).add_node('xgboost').to_pipeline())
    show_pyvis(pipeline)
    show_pyvis_custom_colors(pipeline)
    try:
        import graphviz
        show_graphviz(pipeline)
        show_graphviz_custom_colors(pipeline)
    except ImportError:
        default_log().info('Either Graphviz or pygraphviz is not installed. Skipping visualizations.')


if __name__ == '__main__':
    main()
