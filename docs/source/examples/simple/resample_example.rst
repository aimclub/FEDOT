
.. _resample_example:

Resample Example
================

This example demonstrates the use of two classification pipelines, one with balancing and one without, and includes an optional tuning process. The example can be run with either synthetic data or a real dataset.

.. note::
    This example requires the `Fedot` framework and related libraries.

Step-by-Step Guide
------------------

1. **Importing Necessary Libraries**

   The first block of code imports all the necessary libraries and modules required for the example.

   .. code-block:: python

    from datetime import timedelta
    import numpy as np
    import pandas as pd
    from golem.core.tuning.simultaneous import SimultaneousTuner
    from sklearn.metrics import roc_auc_score as roc_auc
    from sklearn.model_selection import train_test_split
    from examples.simple.classification.classification_pipelines import classification_pipeline_without_balancing, classification_pipeline_with_balancing
    from examples.simple.classification.classification_with_tuning import get_classification_dataset
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.metrics_repository import RegressionMetricsEnum
    from fedot.core.repository.tasks import TaskTypesEnum, Task
    from fedot.core.utils import fedot_project_root

2. **Function Definition**

   The function `run_resample_example` is defined to encapsulate the entire process. It takes two parameters: `path_to_data` (optional, for specifying a path to a real dataset) and `tune` (a boolean indicating whether to perform tuning).

   .. code-block:: python

    def run_resample_example(path_to_data=None, tune=False):
        ...

3. **Data Preparation**

   Depending on whether `path_to_data` is provided, the function either generates synthetic data or loads and processes a real dataset.

   .. code-block:: python

    if path_to_data is None:
        ...
    else:
        ...

4. **Data Analysis**

   The function prints the counts of each class in the training set to show the class distribution.

   .. code-block:: python

    unique_class, counts_class = np.unique(y_train, return_counts=True)
    print(f'Two classes: {unique_class}')
    print(f'{unique_class[0]}: {counts_class[0]}')
    print(f'{unique_class[1]}: {counts_class[1]}')

5. **Task and Input Data Setup**

   A classification task is defined, and input data objects are created for training and prediction.

   .. code-block:: python

    task = Task(TaskTypesEnum.classification)
    train_input = InputData(idx=np.arange(0, len(x_train)), features=x_train, target=y_train, task=task, data_type=DataTypesEnum.table)
    predict_input = InputData(idx=np.arange(0, len(x_test)), features=x_test, target=None, task=task, data_type=DataTypesEnum.table)

6. **Pipeline Execution without Balancing**

   A classification pipeline without balancing is created, fitted, and used to predict the test set. The ROC-AUC score is calculated and printed.

   .. code-block:: python

    print('Begin fit Pipeline without balancing')
    pipeline = classification_pipeline_without_balancing()
    pipeline.fit_from_scratch(train_input)
    predict_labels = pipeline.predict(predict_input)
    preds = predict_labels.predict
    print(f'ROC-AUC of pipeline without balancing {roc_auc(y_test, preds):.4f}\n')

7. **Pipeline Execution with Balancing**

   A classification pipeline with balancing is created, fitted, and used to predict the test set. The ROC-AUC score is calculated and printed.

   .. code-block:: python

    print('Begin fit Pipeline with balancing')
    pipeline = classification_pipeline_with_balancing()
    pipeline.fit(train_input)
    predict_labels = pipeline.predict(predict_input)
    preds = predict_labels.predict
    print(f'ROC-AUC of pipeline with balancing {roc_auc(y_test, preds):.4f}\n')

8. **Tuning Process (Optional)**

   If `tune` is set to True, the function performs a tuning process on the pipeline with balancing. The tuned pipeline is then fitted and used to predict the test set. The ROC-AUC score of the tuned pipeline is calculated and printed.

   .. code-block:: python

    if tune:
        ...
        print(f'ROC-AUC of tuned pipeline with balancing - {roc_auc(y_test, preds_tuned):.4f}\n')

9. **Running the Example**

   The example is run twice: once with synthetic data and once with a real dataset, the latter including the tuning process.

   .. code-block:: python

    if __name__ == '__main__':
        run_resample_example()
        print('=' * 25)
        run_resample_example(f'{fedot_project_root()}/examples/data/credit_card_anomaly.csv', tune=True)

.. note::
    Ensure that the paths to datasets and the `fedot_project_root()` function are correctly configured in your environment.

.. seealso::
    For more detailed information on the `Fedot` framework and its capabilities, refer to the `official documentation <https://fedot.readthedocs.io/>`_.

This documentation page provides a comprehensive overview of the example, breaking down the code into logical blocks and explaining each step. Users should be able to understand and replicate the example with their own data.