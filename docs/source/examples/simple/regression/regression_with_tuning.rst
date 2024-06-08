
.. _regression_with_tuning:

====================================================================
Regression Example with Tuning
====================================================================

This example demonstrates how to use a regression pipeline with tuning capabilities. The pipeline is tested on different datasets with varying numbers of samples, features, and options. The goal is to showcase the pipeline's ability to handle diverse datasets and to optimize its performance using a tuner.

Overview
--------

The example consists of two main functions: `get_regression_dataset` and `run_experiment`. The `get_regression_dataset` function generates a synthetic regression dataset with specified parameters. The `run_experiment` function uses this dataset to train, predict, and tune a regression model using a pipeline.

Step-by-Step Guide
------------------

1. Importing Necessary Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from datetime import timedelta
    import numpy as np
    from golem.core.tuning.sequential import SequentialTuner
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from examples.simple.regression.regression_pipelines import regression_ransac_pipeline
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.metrics_repository import RegressionMetricsEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    from fedot.core.utils import set_random_seed
    from fedot.utilities.synth_dataset_generator import regression_dataset

2. Generating a Regression Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def get_regression_dataset(features_options, samples_amount=250,
                               features_amount=5):
        ...
        return x_train, y_train, x_test, y_test

This function generates a synthetic regression dataset with the specified number of samples and features. It also applies a random scaling factor to each feature and splits the data into training and testing sets.

3. Running the Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def run_experiment(pipeline, tuner):
        ...
        if __name__ == '__main__':
            set_random_seed(2020)
            run_experiment(regression_ransac_pipeline(), tuner=SequentialTuner)

The `run_experiment` function iterates over different configurations of samples, features, and options. For each configuration, it generates a dataset, defines a regression task, trains the pipeline, predicts on the test set, and calculates the mean absolute error (MAE). If a tuner is provided, it also tunes the pipeline and reports the MAE after tuning.

4. Pipeline Tuning
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    if tuner is not None:
        ...
        pipeline_tuner = (
            TunerBuilder(task)
            .with_tuner(tuner)
            .with_metric(RegressionMetricsEnum.MAE)
            .with_iterations(50)
            .with_timeout(timedelta(seconds=50))
            .build(train_input)
        )
        tuned_pipeline = pipeline_tuner.tune(pipeline)
        ...

If a tuner is provided, the pipeline is tuned using the specified metric (MAE), number of iterations, and timeout. The tuned pipeline is then used to predict and calculate the MAE.

Conclusion
----------

This example provides a comprehensive demonstration of how to use a regression pipeline with tuning capabilities. It showcases the pipeline's flexibility in handling different datasets and its ability to improve performance through tuning. Users can easily adapt this example to their own regression tasks by modifying the dataset generation parameters and the tuning configuration.