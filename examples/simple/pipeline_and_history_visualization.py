from pathlib import Path

from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory
from fedot.core.utils import fedot_project_root


def run_pipeline_and_history_visualization():
    """ The function runs visualization of the composing history and the best pipeline. """
    # Gather pipeline and history.
    history = OptHistory.load(Path(fedot_project_root(), 'examples', 'data', 'histories', 'scoring_case_history.json'))
    pipeline = PipelineAdapter().restore(history.individuals[-1][-1].graph)
    # Show visualizations.
    pipeline.show()
    history.show()  # The same as `history.show.fitness_line()`
    history.show.fitness_box(best_fraction=0.5)
    history.show.operations_kde()
    history.show.operations_animated_bar(save_path='example_animation.gif', show_fitness=True)
    history.show.fitness_line_interactive()


if __name__ == '__main__':
    run_pipeline_and_history_visualization()
