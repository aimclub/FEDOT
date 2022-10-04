from fedot.core.dag.graph_utils import distance_to_primary_level
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.fitness.multi_objective_fitness import MultiObjFitness
from fedot.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.pipelines.convert import graph_structure_as_nx_graph, pipeline_template_as_nx_graph
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.template import PipelineTemplate

from fedot.core.visualisation.graph_viz import get_hierarchy_pos
from fedot.core.visualisation.opt_viz_extra import extract_objectives


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

    assert len(graph) == pipeline.length  # check node quantity
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
    real_hierarchy_levels_y = {0: ['logit'],
                               1: ['lda', 'rf'],
                               2: ['rf'],
                               3: ['logit', 'knn'],
                               4: ['lda']}
    real_hierarchy_levels_x = {0: ['logit', 'lda', 'logit', 'lda'],
                               1: ['rf', 'knn'],
                               2: ['rf']}

    graph, node_labels = graph_structure_as_nx_graph(pipeline)
    for n, data in graph.nodes(data=True):
        data['hierarchy_level'] = distance_to_primary_level(node_labels[n])
        node_labels[n] = str(node_labels[n])

    pos, _ = get_hierarchy_pos(graph)
    comparable_lists_y = make_comparable_lists(pos, real_hierarchy_levels_y,
                                               node_labels, 1, reverse=True)
    comparable_lists_x = make_comparable_lists(pos, real_hierarchy_levels_x,
                                               node_labels, 0, reverse=False)
    assert comparable_lists_y[0] == comparable_lists_y[1]  # check nodes hierarchy by y axis
    assert comparable_lists_x[0] == comparable_lists_x[1]  # check nodes hierarchy by x axis


def test_extract_objectives():
    num_of_inds = 5
    opt_graph = PipelineAdapter().adapt(pipeline_first())
    individuals = [Individual(opt_graph) for _ in range(num_of_inds)]
    fitness = (-0.8, 0.1)
    weights = tuple([-1 for _ in range(len(fitness))])
    for ind in individuals:
        ind.set_evaluation_result(MultiObjFitness(values=fitness, weights=weights))
    populations_num = 3
    individuals_history = [individuals for _ in range(populations_num)]
    all_objectives = extract_objectives(individuals=individuals_history, transform_from_minimization=True)
    assert all_objectives[0][0] > 0 and all_objectives[0][2] > 0
    assert all_objectives[1][0] > 0 and all_objectives[1][2] > 0
