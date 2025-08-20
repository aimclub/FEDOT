.. _regression_example:

=========================================================================
Regression Example with Fedot Framework
=========================================================================

This example demonstrates how to use the Fedot framework to perform regression analysis on a dataset. The code provided imports necessary modules, loads data, sets up a regression task, trains a model, makes predictions, and visualizes the results.

.. code-block:: python

    import logging

    from fedot import Fedot
    from fedot.core.data.data import InputData
    from fedot.core.data.data_split import train_test_data_setup
    from fedot.core.repository.tasks import TaskTypesEnum, Task
    from fedot.core.utils import fedot_project_root

    def run_regression_example(visualise: bool = False, with_tuning: bool = True,
                               timeout: float = 2., preset: str = 'auto'):
        data_path = f'{fedot_project_root()}/cases/data/cholesterol/cholesterol.csv'

        data = InputData.from_csv(data_path,
                                  task=Task(TaskTypesEnum.regression))
        train, test = train_test_data_setup(data)
        problem = 'regression'

        composer_params = {'history_dir': 'custom_history_dir', 'preset': preset}
        auto_model = Fedot(problem=problem, seed=42, timeout=timeout, logging_level=logging.FATAL,
                           with_tuning=with_tuning, **composer_params)

        auto_model.fit(features=train, target='target')
        prediction = auto_model.predict(features=test)
        if visualise:
            auto_model.history.save('saved_regression_history.json')
            auto_model.plot_prediction()
        print(auto_model.get_metrics())
        return prediction

    if __name__ == '__main__':
        run_regression_example(visualise=True)

Step-by-Step Guide
------------------

1. **Importing Modules**

   The first block imports necessary modules from the Fedot framework and the Python standard library.

   .. code-block:: python

       import logging

       from fedot import Fedot
       from fedot.core.data.data import InputData
       from fedot.core.data.data_split import train_test_data_setup
       from fedot.core.repository.tasks import TaskTypesEnum, Task
       from fedot.core.utils import fedot_project_root

2. **Defining the Function**

   The function `run_regression_example` is defined with parameters for visualization, model tuning, timeout, and preset configuration.

   .. code-block:: python

       def run_regression_example(visualise: bool = False, with_tuning: bool = True,
                                  timeout: float = 2., preset: str = 'auto'):

3. **Loading and Preparing Data**

   The data is loaded from a CSV file and prepared for the regression task. The data is then split into training and testing sets.

   .. code-block:: python

       data_path = f'{fedot_project_root()}/cases/data/cholesterol/cholesterol.csv'

       data = InputData.from_csv(data_path,
                                 task=Task(TaskTypesEnum.regression))
       train, test = train_test_data_setup(data)
       problem = 'regression'

4. **Configuring and Training the Model**

   A Fedot model is configured with specified parameters and trained on the training data.

   .. code-block:: python

       composer_params = {'history_dir': 'custom_history_dir', 'preset': preset}
       auto_model = Fedot(problem=problem, seed=42, timeout=timeout, logging_level=logging.FATAL,
                          with_tuning=with_tuning, **composer_params)

       auto_model.fit(features=train, target='target')

5. **Making Predictions and Visualizing Results**

   The model makes predictions on the test data. If `visualise` is set to True, the history is saved and a prediction plot is generated.

   .. code-block:: python

       prediction = auto_model.predict(features=test)
       if visualise:
           auto_model.history.save('saved_regression_history.json')
           auto_model.plot_prediction()

6. **Printing Metrics and Returning Prediction**

   The model's metrics are printed, and the prediction results are returned.

   .. code-block:: python

       print(auto_model.get_metrics())
       return prediction

7. **Running the Example**

   The example is executed with visualization enabled.

   .. code-block:: python

       if __name__ == '__main__':
           run_regression_example(visualise=True)

This documentation page provides a comprehensive overview of the regression example using the Fedot framework. Users can copy and paste the provided code to apply regression analysis to their own datasets.