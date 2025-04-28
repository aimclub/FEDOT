
.. _fedot_classification_example:

====================================================================
Classification Example with FEDOT Framework
====================================================================

This example demonstrates how to use the FEDOT (Framework for Evidential Data Transformation) to perform a classification task. The example uses a predefined dataset for training and testing, and it showcases the setup and execution of a classification pipeline with hyperparameter tuning and evaluation metrics.

Overview
--------

The FEDOT framework is designed to automate the process of building and optimizing machine learning pipelines. This example specifically focuses on a classification problem, where the goal is to predict a categorical target variable based on input features.

The example is structured into several logical blocks:

1. **Initialization and Configuration**: Setting up the FEDOT instance with the desired problem type, configuration options, and evaluation metrics.
2. **Data Loading**: Specifying the paths to the training and testing datasets.
3. **Model Training**: Fitting the FEDOT pipeline to the training data.
4. **Prediction**: Generating probability predictions on the test data.
5. **Visualization**: Plotting the prediction results.

Step-by-Step Guide
------------------

Below is a detailed breakdown of the code, ensuring that each line is explained and can be easily understood and replicated.

Initialization and Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from fedot import FedotBuilder
    from fedot.core.utils import fedot_project_root

    if __name__ == '__main__':
        train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
        test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

        fedot = (FedotBuilder(problem='classification')
                 .setup_composition(timeout=10, with_tuning=True, preset='best_quality')
                 .setup_pipeline_evaluation(max_pipeline_fit_time=5, metric=['roc_auc', 'precision'])
                 .build())

In this block, the FEDOT framework is imported, and the paths to the training and testing datasets are defined. The FEDOT instance is then configured for a classification problem, with a timeout for the composition process, hyperparameter tuning enabled, and a preset for the best quality. The evaluation metrics are set to `roc_auc` and `precision`.

Data Loading
^^^^^^^^^^^^

The data loading is implicit in the paths defined in the initialization block. The paths are constructed using the `fedot_project_root()` function, which returns the root directory of the FEDOT project, and then the paths to the specific CSV files are appended.

Model Training
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    fedot.fit(features=train_data_path, target='target')

This line fits the FEDOT pipeline to the training data. The `features` parameter is set to the path of the training dataset, and the `target` parameter specifies the name of the target column in the dataset.

Prediction
^^^^^^^^^^

.. code-block:: python

    fedot.predict_proba(features=test_data_path)

Here, the FEDOT pipeline is used to generate probability predictions for the test dataset. The `features` parameter is set to the path of the test dataset.

Visualization
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    fedot.plot_prediction()

The final line of the example plots the prediction results, providing a visual representation of the model's performance.

Conclusion
----------

This example provides a comprehensive guide on how to use the FEDOT framework for a classification task. By following this guide, users can replicate the example and adapt it to their own datasets and classification problems.

.. note::
    Ensure that the required datasets are available at the specified paths and that the FEDOT framework is properly installed and configured.

.. seealso::
    For more information on the FEDOT framework, visit the `official documentation <https://github.com/nccr-itmo/FEDOT>`_.

This documentation page is formatted in .rst (reStructuredText) for use in Sphinx-based documentation systems, which are commonly used for Python projects. It provides a clear and structured explanation of the code example, ensuring that users can understand and apply the example to their own classification tasks.