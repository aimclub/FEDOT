from examples.simple.classification.classification_pipelines import classification_xgboost_complex_pipeline
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.visualisation.opt_viz import PipelineEvolutionVisualiser


def generate_history(generations, pop_size, folder=None):
    history = OptHistory(save_folder=folder)
    converter = GraphGenerationParams().adapter
    for gen in range(generations):
        new_pop = []
        for idx in range(pop_size):
            pipeline = classification_xgboost_complex_pipeline()
            ind = Individual(converter.adapt(pipeline))
            ind.fitness = 1 / (gen * idx + 1)
            new_pop.append(ind)
        history.add_to_history(new_pop)
    return history


def run_pipeline_ang_history_visualisation(generations=2, pop_size=10,
                                           with_pipeline_visualisation=True):
    """ Function run visualisation of composing history and pipeline """
    # Generate pipeline and history
    pipeline = classification_xgboost_complex_pipeline()
    history = generate_history(generations, pop_size)

    visualiser = PipelineEvolutionVisualiser()
    visualiser.visualise_history(history)
    if with_pipeline_visualisation:
        pipeline.show()


if __name__ == '__main__':
    run_pipeline_ang_history_visualisation()
