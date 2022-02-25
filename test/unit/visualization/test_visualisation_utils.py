from fedot.core.optimisers.utils.multi_objective_fitness import MultiObjFitness
from fedot.core.pipelines.convert import pipeline_template_as_nx_graph
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.visualisation.graph_viz import hierarchy_pos
from fedot.core.visualisation.opt_viz import PipelineEvolutionVisualiser


def pipeline_first():  # tested pipeline
    #    XG
    #  |     \
    # XG     KNN
    # |  \    |  \
    # LR LDA LR  LDA
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


def test_pipeline_template_as_nx_graph():
    pipeline = pipeline_first()
    pipeline_template = PipelineTemplate(pipeline)
    graph, node_labels = pipeline_template_as_nx_graph(pipeline=pipeline_template)

    assert len(graph.nodes) == len(pipeline.nodes)  # check node quantity
    assert node_labels[0] == str(pipeline.root_node)  # check root node


def make_comparable_lists(pos, real_hierarchy_levels, node_labels, dim, reverse):
    def extract_levels(hierarchy_levels):
        levels = []
        for pair in hierarchy_levels:
            levels.append(sorted(pair[1]))
        return levels

    computed_hierarchy_levels = {}
    for node in pos:
        level = pos[node][dim]
        if level in computed_hierarchy_levels:
            computed_hierarchy_levels[level].append(node_labels[node])
        else:
            computed_hierarchy_levels[level] = [node_labels[node]]

    sorted_computed_hierarchy_levels = sorted(computed_hierarchy_levels.items(),
                                              key=lambda x: x[0], reverse=reverse)
    sorted_real_hierarchy_levels = sorted(real_hierarchy_levels.items(),
                                          key=lambda x: x[0])
    return extract_levels(sorted_computed_hierarchy_levels), extract_levels(sorted_real_hierarchy_levels)


def test_hierarchy_pos():
    pipeline = pipeline_first()
    real_hierarchy_levels_y = {0: ['rf'], 1: ['rf', 'knn'],
                               2: ['logit', 'lda', 'logit', 'lda']}
    real_hierarchy_levels_x = {0: ['logit'], 1: ['rf'], 2: ['lda'],
                               3: ['rf'], 4: ['logit'], 5: ['knn'], 6: ['lda']}
    pipeline_template = PipelineTemplate(pipeline)
    graph, node_labels = pipeline_template_as_nx_graph(pipeline=pipeline_template)
    pos = hierarchy_pos(graph.to_undirected(), root=0)
    comparable_lists_y = make_comparable_lists(pos, real_hierarchy_levels_y,
                                               node_labels, 1, reverse=True)
    comparable_lists_x = make_comparable_lists(pos, real_hierarchy_levels_x,
                                               node_labels, 0, reverse=False)
    assert comparable_lists_y[0] == comparable_lists_y[1]  # check nodes hierarchy by y axis
    assert comparable_lists_x[0] == comparable_lists_x[1]  # check nodes hierarchy by x axis


def test_extract_objectives():
    visualiser = PipelineEvolutionVisualiser()
    num_of_inds = 5
    individuals = [pipeline_first() for _ in range(num_of_inds)]
    fitness = (-0.8, 0.1)
    weights = tuple([-1 for _ in range(len(fitness))])
    for ind in individuals:
        ind.fitness = MultiObjFitness(values=fitness, weights=weights)
    populations_num = 3
    individuals_history = [individuals for _ in range(populations_num)]
    all_objectives = visualiser.extract_objectives(individuals=individuals_history, transform_from_minimization=True)
    assert all_objectives[0][0] > 0 and all_objectives[0][2] > 0
    assert all_objectives[1][0] > 0 and all_objectives[1][2] > 0
