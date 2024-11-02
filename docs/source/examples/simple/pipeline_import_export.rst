.. _import_export_example:

Import and Export of Regression Pipeline Example
===========================================================================

This example demonstrates how to import and export a regression pipeline using the Fedot framework. The pipeline is specifically designed for regression tasks and uses the RANSAC algorithm for model fitting. The example covers the creation of a regression dataset, defining the regression task, training the model, and then exporting and importing the pipeline to verify its functionality.

.. code-block:: python

    import json
    import os

    import numpy as np

    from examples.simple.regression.regression_with_tuning import get_regression_dataset
    from examples.simple.regression.regression_pipelines import regression_ransac_pipeline
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    from fedot.core.utils import fedot_project_root

    def create_correct_path(path: str, dirname_flag: bool = False):
        """
        Create path with time which was created during the testing process.
        """
        # TODO: this function is used in many places, but now is not really needed
        last_el = None
        for dirname in next(os.walk(os.path.curdir))[1]:
            if dirname.endswith(path):
                if dirname_flag:
                    last_el = dirname
                else:
                    file = os.path.join(dirname, path + '.json')
                    last_el = file
        return last_el

    def run_import_export_example(pipeline_path, pipeline):
        features_options = {'informative': 1, 'bias': 0.0}
        samples_amount = 100
        features_amount = 2
        x_train, y_train, x_test, y_test = get_regression_dataset(features_options,
                                                                  samples_amount,
                                                                  features_amount)

        # Define regression task
        task = Task(TaskTypesEnum.regression)

        # Prepare data to train the model
        train_input = InputData(idx=np.arange(0, len(x_train)),
                                features=x_train,
                                target=y_train,
                                task=task,
                                data_type=DataTypesEnum.table)

        predict_input = InputData(idx=np.arange(0, len(x_test)),
                                  features=x_test,
                                  target=None,
                                  task=task,
                                  data_type=DataTypesEnum.table)

        # Get pipeline and fit it
        pipeline.fit_from_scratch(train_input)

        predicted_output = pipeline.predict(predict_input)
        prediction_before_export = np.array(predicted_output.predict)
        print(f'Before export {prediction_before_export[:4]}')

        # Export it
        path_to_save_and_load = f'{fedot_project_root()}/examples/simple/{pipeline_path}'
        pipeline.save(path=path_to_save_and_load, create_subdir=False, is_datetime_in_path=False)

        # Import pipeline
        new_pipeline = Pipeline().load(path_to_save_and_load)

        predicted_output_after_export = new_pipeline.predict(predict_input)
        prediction_after_export = np.array(predicted_output_after_export.predict)

        print(f'After import {prediction_after_export[:4]}')

        dict_pipeline, dict_fitted_operations = pipeline.save()
        dict_pipeline = json.loads(dict_pipeline)
        pipeline_from_dict = Pipeline.from_serialized(dict_pipeline, dict_fitted_operations)

        predicted_output = pipeline_from_dict.predict(predict_input)
        prediction = np.array(predicted_output.predict)
        print(f'Prediction from pipeline loaded from dict {prediction[:4]}')

    if __name__ == '__main__':
        run_import_export_example(pipeline_path='import_export', pipeline=regression_ransac_pipeline())

Step-by-Step Guide
------------------

1. **Import Necessary Libraries and Modules**

   The example starts by importing necessary libraries and modules required for the regression task and pipeline operations.

   .. code-block:: python

       import json
       import os

       import numpy as np

       from examples.simple.regression.regression_with_tuning import get_regression_dataset
       from examples.simple.regression.regression_pipelines import regression_ransac_pipeline
       from fedot.core.data.data import InputData
       from fedot.core.pipelines.pipeline import Pipeline
       from fedot.core.repository.dataset_types import DataTypesEnum
       from fedot.core.repository.tasks import Task, TaskTypesEnum
       from fedot.core.utils import fedot_project_root

2. **Create a Function to Handle Paths**

   A function `create_correct_path` is defined to handle paths, although it is commented as not needed for the current example.

   .. code-block:: python

       def create_correct_path(path: str, dirname_flag: bool = False):
           # ...

3. **Define the Main Function for Import and Export**

   The `run_import_export_example` function is defined to handle the entire process of creating a dataset, training a model, and then exporting and importing the pipeline.

   .. code-block:: python

       def run_import_export_example(pipeline_path, pipeline):
           # ...

4. **Create a Regression Dataset**

   The `get_regression_dataset` function is used to generate a dataset with specified parameters.

   .. code-block:: python

       x_train, y_train, x_test, y_test = get_regression_dataset(features_options,
                                                                 samples_amount,
                                                                 features_amount)

5. **Define the Regression Task**

   A regression task is defined using the `Task` class from the Fedot framework.

   .. code-block:: python

       task = Task(TaskTypesEnum.regression)

6. **Prepare Data for Training and Prediction**

   Data is prepared for both training and prediction using the `InputData` class.

   .. code-block:: python

       train_input = InputData(idx=np.arange(0, len(x_train)),
                               features=x_train,
                               target=y_train,
                               task=task,
                               data_type=DataTypesEnum.table)

       predict_input = InputData(idx=np.arange(0, len(x_test)),
                                 features=x_test,
                                 target=None,
                                 task=task,
                                 data_type=DataTypesEnum.table)

7. **Train the Pipeline**

   The pipeline is trained using the training data.

   .. code-block:: python

       pipeline.fit_from_scratch(train_input)

8. **Predict Using the Trained Pipeline**

   Predictions are made on the test data using the trained pipeline.

   .. code-block:: python

       predicted_output = pipeline.predict(predict_input)

9. **Export the Pipeline**

   The pipeline is exported to a specified path.

   .. code-block:: python

       pipeline.save(path=path_to_save_and_load, create_subdir=False, is_datetime_in_path=False)

10. **Import the Pipeline**

    The pipeline is imported from the saved path.

    .. code-block:: python

        new_pipeline = Pipeline().load(path_to_save_and_load)

11. **Predict Using the Imported Pipeline**

    Predictions are made again to verify the functionality of the imported pipeline.

    .. code-block:: python

        predicted_output_after_export = new_pipeline.predict(predict_input)

12. **Save and Load Pipeline from Dictionary**

    The pipeline is also saved and loaded from a dictionary format to demonstrate an alternative method of serialization.

    .. code-block:: python

        dict_pipeline, dict_fitted_operations = pipeline.save()
        dict_pipeline = json.loads(dict_pipeline)
        pipeline_from_dict = Pipeline.from_serialized(dict_pipeline, dict_fitted_operations)

13. **Execute the Main Function**

    The `run_import_export_example` function is executed with a specified pipeline path and the regression pipeline.

    .. code-block:: python

        if __name__ == '__main__':
            run_import_export_example(pipeline_path='import_export', pipeline=regression_ransac_pipeline())

This documentation page provides a comprehensive guide to understanding and using the import and export functionality of regression pipelines in the Fedot framework. Users can copy and paste the provided code snippets to implement similar functionality in their own projects.