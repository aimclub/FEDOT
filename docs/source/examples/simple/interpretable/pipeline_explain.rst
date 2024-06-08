
.. _pipeline_explain_example:

=========================================================================
Example: Explaining a Classification Pipeline
=========================================================================

This example demonstrates how to use the `Fedot` framework to explain a classification pipeline. The pipeline is applied to a dataset from a CSV file, and the explanation of the pipeline's decision-making process is visualized.

Overview
--------

The example script performs the following tasks:

1. Loads training data from a CSV file.
2. Constructs a complex classification pipeline using predefined components.
3. Fits the pipeline to the training data.
4. Explains the pipeline using a surrogate decision tree model.
5. Visualizes the explanation and saves the plot.

Step-by-Step Guide
------------------

1. **Specifying Paths**

   The paths for the training data and the output figure are specified:

   .. code-block:: python

      train_data_path = os.path.join(fedot_project_root(), 'cases', 'data', 'cancer', 'cancer_train.csv')
      figure_path = 'pipeline_explain_example.png'

2. **Feature and Class Names for Visualization**

   The feature and class names are extracted from the training data:

   .. code-block:: python

      feature_names = pd.read_csv(train_data_path, index_col=0, nrows=0).columns.tolist()
      target_name = feature_names.pop()
      target = pd.read_csv(train_data_path, usecols=[target_name])[target_name]
      class_names = target.unique().astype(str).tolist()

3. **Data Load**

   The training data is loaded into an `InputData` object:

   .. code-block:: python

      train_data = InputData.from_csv(train_data_path)

4. **Pipeline Composition**

   A complex classification pipeline is composed using a predefined function:

   .. code-block:: python

      pipeline = classification_complex_pipeline()

5. **Pipeline Fitting**

   The pipeline is fitted to the training data:

   .. code-block:: python

      pipeline.fit(train_data)

6. **Pipeline Explaining**

   The pipeline is explained using a surrogate decision tree model:

   .. code-block:: python

      explainer = explain_pipeline(pipeline, data=train_data, method='surrogate_dt', visualization=True)

7. **Visualizing Explanation and Saving the Plot**

   The explanation is visualized and the plot is saved:

   .. code-block:: python

      print(f'Built surrogate model: {explainer.surrogate_str}')
      explainer.visualize(save_path=figure_path, dpi=200, feature_names=feature_names, class_names=class_names,
                          precision=6)

Running the Example
-------------------

To run this example, execute the following code:

.. code-block:: python

   if __name__ == '__main__':
       run_pipeline_explain()

This will execute the `run_pipeline_explain` function, which performs all the steps described above.

.. note::
   Ensure that the required data files and dependencies are available in your environment before running the example.

.. seealso::
   For more information on the `Fedot` framework and its capabilities, refer to the `official documentation <https://github.com/nccr-itmo/FEDOT>`_.
