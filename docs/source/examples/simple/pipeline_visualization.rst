
.. _pipeline_visualization_example:

=========================================================================
Example: Visualizing Machine Learning Pipelines
=========================================================================

This example demonstrates how to create and visualize a machine learning pipeline using the Fedot framework. The pipeline includes preprocessing steps and multiple models for prediction. The visualization options include default settings, customized styles, and different rendering engines such as networkx, pyvis, and graphviz.

.. note::
   This example requires the `Fedot` library and optionally `pyvis` and `graphviz` for advanced visualization.

.. contents:: Table of Contents
   :depth: 2
   :local:

Creating the Pipeline
---------------------

The first step is to define the structure of the pipeline. This is done by specifying the nodes and their dependencies.

.. code-block:: python

   from fedot.core.pipelines.node import PipelineNode
   from fedot.core.pipelines.pipeline import Pipeline

   def generate_pipeline() -> Pipeline:
       node_scaling = PipelineNode('scaling')
       node_first = PipelineNode('kmeans', nodes_from=[node_scaling])
       node_second = PipelineNode('rf', nodes_from=[node_scaling])
       node_third = PipelineNode('linear', nodes_from=[node_scaling])
       node_root = PipelineNode('logit', nodes_from=[node_first, node_second, node_third, node_scaling])

       return Pipeline(node_root)

Visualizing the Pipeline
-------------------------

The pipeline can be visualized using different methods and customization options.

Default Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def show_default(pipeline: Pipeline):
       """ Show with default properties via networkx. """
       pipeline.show()

Customized Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def show_customized(pipeline: Pipeline):
       """ Show with adjusted sizes and green nodes. """
       pipeline.show(node_color='green', edge_curvature_scale=1.2, node_size_scale=2.0, font_size_scale=1.4)

Custom Colors Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def show_custom_colors(pipeline: Pipeline):
       """ Show with colors defined by label-color dictionary. """
       pipeline.show(node_color={'scaling': 'tab:olive', 'rf': '#FF7F50', 'linear': (0, 1, 1), None: 'black'},
                     edge_curvature_scale=1.2, node_size_scale=2.0, font_size_scale=1.4)

Function-Defined Colors Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def show_complex_colors(pipeline: Pipeline):
       """ Show with colors defined by function. """
       def nodes_color(labels):
           if 'xgboost' in labels:
               return {'xgboost': 'tab:orange', None: 'black'}
           else:
               return {'rf': 'tab:green', None: 'black'}

       pipeline.show(node_color=nodes_color)

Pyvis Visualization
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def show_pyvis(pipeline: Pipeline):
       """ Show with pyvis. """
       pipeline.show(engine='pyvis')

   def show_pyvis_custom_colors(pipeline: Pipeline):
       """ Show with pyvis with custom colors. """
       pipeline.show(engine='pyvis',
                     node_color={'scaling': 'tab:olive', 'rf': '#FF7F50', 'linear': (0, 1, 1), None: 'black'})

Graphviz Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def show_graphviz(pipeline: Pipeline):
       """ Show with graphviz (requires Graphviz and pygraphviz). """
       pipeline.show(engine='graphviz')

   def show_graphviz_custom_colors(pipeline: Pipeline):
       """ Show with graphviz with custom colors (requires Graphviz and pygraphviz). """
       pipeline.show(engine='graphviz',
                     node_color={'scaling': 'tab:olive', 'rf': '#FF7F50', 'linear': (0, 1, 1), None: 'black'})

Running the Example
--------------------

The main function orchestrates the creation of the pipeline and its visualization using various methods.

.. code-block:: python

   def main():
       pipeline = generate_pipeline()
       show_default(pipeline)
       show_customized(pipeline)
       show_custom_colors(pipeline)
       show_complex_colors(pipeline)
       show_complex_colors(PipelineBuilder(*pipeline.nodes).add_node('xgboost').build())
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

.. note::
   Ensure that you have the necessary libraries installed and configured correctly for the visualization engines you wish to use.

.. seealso::
   - `Fedot Documentation <https://fedot.readthedocs.io>`_
   - `Pyvis Documentation <https://pyvis.readthedocs.io>`_
   - `Graphviz Documentation <https://graphviz.readthedocs.io>`_
