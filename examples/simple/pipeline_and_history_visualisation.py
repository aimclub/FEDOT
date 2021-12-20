from fedot.core.optimisers.gp_comp.gp_optimiser import GraphGenerationParams
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.visualisation.opt_viz import PipelineEvolutionVisualiser


def pipeline_first():
    #    XG
    #  |     \
    # XG     KNN
    # |  \    |  \
    # LR LDA LR  LDA
    pipeline = Pipeline()

    root_of_tree, root_child_first, root_child_second = \
        [SecondaryNode(model) for model in ('xgboost', 'xgboost', 'knn')]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in ('logit', 'lda'):
            new_node = PrimaryNode(requirement_model)
            root_node_child.nodes_from.append(new_node)
            pipeline.add_node(new_node)
        pipeline.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    pipeline.add_node(root_of_tree)
    return pipeline


def generate_history(generations, pop_size, folder=None):
    history = OptHistory(save_folder=folder)
    converter = GraphGenerationParams().adapter
    for gen in range(generations):
        new_pop = []
        for idx in range(pop_size):
            pipeline = pipeline_first()
            ind = Individual(converter.adapt(pipeline))
            ind.fitness = 1 / (gen * idx + 1)
            new_pop.append(ind)
        history.add_to_history(new_pop)
    return history


def run_pipeline_ang_history_visualisation(generations=2, pop_size=10,
                                           with_pipeline_visualisation=True):
    """ Function run visualisation of composing history and pipeline """
    # Generate pipeline and history
    pipeline = pipeline_first()
    history = generate_history(generations, pop_size)

    visualiser = PipelineEvolutionVisualiser()
    visualiser.visualise_history(history)
    if with_pipeline_visualisation:
        pipeline.show()


if __name__ == '__main__':
    run_pipeline_ang_history_visualisation()
