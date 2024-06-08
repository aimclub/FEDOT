
.. _classification_tuning_example:

====================================================================
Classification Tuning Example with Fedot
====================================================================

This example demonstrates how to use the Fedot framework for tuning a classification pipeline. The example generates synthetic classification datasets with varying parameters and applies a Random Forest classifier, which is then optionally tuned using a simultaneous tuner.

.. note::
    This example requires the `Fedot` framework and its dependencies to be installed.

.. contents:: Table of Contents
    :depth: 2
    :local:

Setup and Imports
-----------------

The first step is to import necessary modules and libraries:

.. code-block:: python

    import numpy as np
    from golem.core.tuning.simultaneous import SimultaneousTuner
    from sklearn.metrics import roc_auc_score as roc_auc
    from sklearn.model_selection import train_test_split

    from examples.simple.classification.classification_pipelines import classification_random_forest_pipeline
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    from fedot.core.utils import set_random_seed
    from fedot.utilities.synth_dataset_generator import classification_dataset

Data Preparation
----------------

The `get_classification_dataset` function prepares synthetic classification datasets with specified parameters:

.. code-block:: python

    def get_classification_dataset(features_options, samples_amount=250,
                                   features_amount=5, classes_amount=2, weights=None):
        ...
        return x_data_train, y_data_train, x_data_test, y_data_test

Model Prediction Conversion
---------------------------

The `convert_to_labels` function converts model predictions to binary labels:

.. code-block:: python

    def convert_to_labels(root_operation, prediction):
        ...
        return preds

Classification Tuning Experiment
--------------------------------

The `run_classification_tuning_experiment` function runs the classification tuning experiment:

.. code-block:: python

    def run_classification_tuning_experiment(pipeline, tuner=None):
        ...
        if tuner is not None:
            ...
            print('Obtained metrics after tuning:')
            print(f"{roc_auc(y_test, preds_tuned):.4f}\n")

Running the Example
-------------------

To run the example, execute the following script:

.. code-block:: python

    if __name__ == '__main__':
        set_random_seed(2020)
        run_classification_tuning_experiment(pipeline=classification_random_forest_pipeline(),
                                             tuner=SimultaneousTuner)

.. note::
    Ensure you have the necessary permissions and dependencies installed to run the script.

Conclusion
----------

This example showcases the use of Fedot for tuning a classification pipeline, demonstrating how to generate synthetic datasets, apply a classifier, and optionally tune the pipeline for better performance.