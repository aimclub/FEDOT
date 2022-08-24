from pathlib import Path

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.utils import fedot_project_root


def run_pipeline_and_history_visualization(with_pipeline_visualization=True):
    """ Function run visualization of composing history and pipeline """
    # Generate pipeline and history
    history = OptHistory.load(Path(fedot_project_root(), 'examples', 'data', 'history', 'opt_history.json'))
    pipeline = PipelineAdapter().restore(history.individuals[-1][-1].graph)

    history.show('fitness_line')
    history.show.fitness_box(best_fraction=0.5)
    history.show('operations_kde')
    history.show('operations_animated_bar', save_path='example_animation.gif', show_fitness=True)
    if with_pipeline_visualization:
        pipeline.show()
    history.show('fitness_line_interactive')


if __name__ == '__main__':
    run_pipeline_and_history_visualization()
