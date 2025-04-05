
.. _tune_pipeline_example:

=========================================================================
Tuning a Machine Learning Pipeline Example
=========================================================================

This example demonstrates how to tune a machine learning pipeline using the Fedot framework. The pipeline is tuned to improve its performance on a regression task using a dataset from a CSV file. The tuning process involves optimizing the pipeline's hyperparameters to minimize the Mean Squared Error (MSE) on a test dataset.

.. note::
    This example requires the Fedot framework to be installed. You can install it using pip:

    .. code-block:: bash

        pip install fedot

Example Overview
================

The example is structured into several logical blocks:

1. **Pipeline Initialization**: A pipeline is created with a decision tree regression model and a KNN regression model.
2. **Data Loading and Splitting**: The dataset is loaded from a CSV file and split into training and testing sets.
3. **Pipeline Tuning**: The pipeline is tuned using an optimization algorithm to find the best hyperparameters.
4. **Evaluation**: The performance of the pipeline before and after tuning is evaluated and compared.

Step-by-Step Guide
==================

1. **Pipeline Initialization**

   The pipeline is initialized using the `PipelineBuilder` class. Nodes for decision tree regression ('dtreg') and KNN regression ('knnreg') are added, and their outputs are joined using a random forest regression ('rfr') node.

   .. code-block:: python

       pipeline = (PipelineBuilder()
                   .add_node('dtreg', 0)
                   .add_node('knnreg', 1)
                   .join_branches('rfr')
                   .build())

2. **Data Loading and Splitting**

   The dataset is loaded from a CSV file and converted into an `InputData` object. The data is then split into training and testing sets.

   .. code-block:: python

       data_path = f'{fedot_project_root()}/cases/data/cholesterol/cholesterol.csv'
       data = InputData.from_csv(data_path, task=Task(TaskTypesEnum.regression))
       train_data, test_data = train_test_data_setup(data)

3. **Pipeline Tuning**

   The `tune_pipeline` function is defined to perform the tuning process. It takes the pipeline, training data, testing data, and the number of tuning iterations as inputs.

   .. code-block:: python

       def tune_pipeline(pipeline: Pipeline,
                         train_data: InputData,
                         test_data: InputData,
                         tuner_iter_num: int = 100):
           ...

   Inside the function, the pipeline is fitted to the training data, and its performance is evaluated on the test data before tuning. Then, a tuner is built using the `TunerBuilder` class, which configures the optimization process.

   .. code-block:: python

       pipeline_tuner = TunerBuilder(task) \
           .with_tuner(IOptTuner) \
           .with_requirements(requirements) \
           .with_metric(metric) \
           .with_iterations(tuner_iter_num) \
           .with_additional_params(eps=0.02, r=1.5, refine_solution=True) \
           .build(train_data)

   The pipeline is then tuned using the tuner, and the tuned pipeline is fitted to the training data.

   .. code-block:: python

       tuned_pipeline = pipeline_tuner.tune(pipeline)
       tuned_pipeline.fit(train_data)

4. **Evaluation**

   The performance of the pipeline after tuning is evaluated on the test data, and the results are printed.

   .. code-block:: python

       after_tuning_predicted = tuned_pipeline.predict(test_data)
       metric_after_tuning = MSE().metric(test_data, after_tuning_predicted)

       print(f'\nMetric before tuning: {metric_before_tuning}')
       print(f'Metric after tuning: {metric_after_tuning}')

   The tuned pipeline is returned by the function.

Running the Example
===================

To run the example, execute the following code:

.. code-block:: python

    if __name__ == '__main__':
        pipeline = (PipelineBuilder()
                    .add_node('dtreg', 0)
                    .add_node('knnreg', 1)
                    .join_branches('rfr')
                    .build())
        data_path = f'{fedot_project_root()}/cases/data/cholesterol/cholesterol.csv'

        data = InputData.from_csv(data_path,
                                  task=Task(TaskTypesEnum.regression))
        train_data, test_data = train_test_data_setup(data)
        tuned_pipeline = tune_pipeline(pipeline, train_data, test_data, tuner_iter_num=200)

This will load the dataset, create and tune the pipeline, and print the MSE before and after tuning.

.. note::
    Make sure to adjust the path to the CSV file if it's located in a different directory.

Conclusion
==========

This example provides a practical demonstration of how to tune a machine learning pipeline using the Fedot framework. By following this guide, users can understand the process of optimizing a pipeline for better performance on regression tasks.