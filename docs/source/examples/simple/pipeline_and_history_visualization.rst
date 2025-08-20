.. _pipeline_and_history_visualization:

=========================================================================
Visualizing Pipeline Composition History Example
=========================================================================

This example demonstrates how to visualize the composition history of a machine learning pipeline and the best pipeline itself. The code provided loads a pipeline optimization history, restores the best pipeline from that history, and then visualizes various aspects of the history and the pipeline.

.. code-block:: python

    from pathlib import Path
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from fedot.core.pipelines.adapters import PipelineAdapter
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

Step-by-Step Guide
------------------

1. **Importing Necessary Modules**

   .. code-block:: python

       from pathlib import Path
       from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
       from fedot.core.pipelines.adapters import PipelineAdapter
       from fedot.core.utils import fedot_project_root
       from fedot.core.visualisation.pipeline_specific_visuals import PipelineHistoryVisualizer

   Here, the necessary modules are imported to handle the pipeline history, restore the pipeline, and visualize the history and pipeline.

2. **Function Definition**

   .. code-block:: python

       def run_pipeline_and_history_visualization():
           """ The function runs visualization of the composing history and the best pipeline. """

   The function `run_pipeline_and_history_visualization` is defined to encapsulate the logic of loading and visualizing the pipeline and its history.

3. **Loading the Pipeline History**

   .. code-block:: python

       history = OptHistory.load(Path(fedot_project_root(), 'examples', 'data', 'histories', 'scoring_case_history.json'))

   The pipeline optimization history is loaded from a JSON file located in the specified path.

4. **Restoring the Best Pipeline**

   .. code-block:: python

       pipeline = PipelineAdapter().restore(history.individuals[-1][-1].graph)

   The best pipeline is restored from the graph representation stored in the last individual of the history.

5. **Visualizing the Pipeline**

   .. code-block:: python

       pipeline.show()

   The pipeline is visualized using its `show` method.

6. **Creating a Visualizer for the History**

   .. code-block:: python

       history_visualizer = PipelineHistoryVisualizer(history)

   A visualizer object is created to handle the visualization of the pipeline history.

7. **Visualizing Different Aspects of the History**

   .. code-block:: python

       history_visualizer.fitness_line()
       history_visualizer.fitness_box(best_fraction=0.5)
       history_visualizer.operations_kde()
       history_visualizer.operations_animated_bar(save_path='example_animation.gif', show_fitness=True)
       history_visualizer.fitness_line_interactive()

   Various methods are called on the visualizer to display different visualizations of the history, including a line plot of fitness, a box plot, a kernel density estimation plot for operations, an animated bar chart, and an interactive line plot of fitness.

8. **Running the Function**

   .. code-block:: python

       if __name__ == '__main__':
           run_pipeline_and_history_visualization()

   The function is called if the script is run as the main program.

This documentation page provides a comprehensive guide to understanding and using the provided code example for visualizing a machine learning pipeline and its composition history. Users can copy and paste the code into their environment and adapt it to their own purposes.