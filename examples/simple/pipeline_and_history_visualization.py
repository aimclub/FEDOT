from pathlib import Path

from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory
from fedot.core.utils import fedot_project_root
from fedot.core.visualisation.pipeline_specific_visuals import PipelineHistoryVisualizer


def run_pipeline_and_history_visualization():
    """ The function runs visualization of the composing history and the best pipeline. """
    # Gather pipeline and history.
    history = OptHistory.load(Path(fedot_project_root(), 'examples', 'data', 'histories', 'scoring_case_history.json'))
    pipeline = PipelineAdapter().restore(history.individuals[-1][-1].graph)
    # Show visualizations.
    pipeline.show()
    history_visualizer = PipelineHistoryVisualizer(history)
    history_visualizer.fitness_line()
    history_visualizer.fitness_box(best_fraction=0.5)
    history_visualizer.operations_kde()
    history_visualizer.operations_animated_bar(save_path='example_animation.gif', show_fitness=True)
    history_visualizer.fitness_line_interactive()


if __name__ == '__main__':
    run_pipeline_and_history_visualization()
