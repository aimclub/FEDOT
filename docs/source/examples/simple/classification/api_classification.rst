
Fedot Classification Example
============================

This example demonstrates how to use the Fedot framework for automated machine learning (AutoML) to perform classification tasks. It compares a baseline model with a more sophisticated automated model that includes hyperparameter tuning.

Overview
--------

The example uses the Fedot library to train and evaluate classification models on a dataset. It first sets up a baseline model using a predefined Random Forest model and then sets up an automated model with hyperparameter tuning. The performance of both models is evaluated, and the automated model's predictions are visualized if the `visualization` parameter is set to `True`.

Step-by-Step Guide
------------------

1. **Import Necessary Modules**

   .. code-block:: python

      from fedot import Fedot
      from fedot.core.utils import fedot_project_root, set_random_seed

2. **Define the `run_classification_example` Function**

   This function takes three parameters: `timeout`, `visualization`, and `with_tuning`. It sets up the problem type, data paths, and initializes the models.

   .. code-block:: python

      def run_classification_example(timeout: float = None, visualization=False, with_tuning=True):
          problem = 'classification'
          train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
          test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

3. **Initialize and Train the Baseline Model**

   The baseline model is initialized with a predefined Random Forest model and trained on the training data.

   .. code-block:: python

          baseline_model = Fedot(problem=problem, timeout=timeout)
          baseline_model.fit(features=train_data_path, target='target', predefined_model='rf')

4. **Make Predictions with the Baseline Model**

   The baseline model makes predictions on the test data and its metrics are printed.

   .. code-block:: python

          baseline_model.predict(features=test_data_path)
          print(baseline_model.get_metrics())

5. **Initialize and Train the Automated Model**

   The automated model is initialized with settings for best quality, hyperparameter tuning, and other parameters. It is trained on the training data.

   .. code-block:: python

          auto_model = Fedot(problem=problem, timeout=timeout, n_jobs=-1, preset='best_quality',
                             max_pipeline_fit_time=5, metric=['roc_auc', 'precision'], with_tuning=with_tuning)
          auto_model.fit(features=train_data_path, target='target')

6. **Make Predictions with the Automated Model**

   The automated model makes probability predictions on the test data, and its metrics are printed with a specified rounding order.

   .. code-block:: python

          prediction = auto_model.predict_proba(features=test_data_path)
          print(auto_model.get_metrics(rounding_order=4))

7. **Visualize the Predictions (Optional)**

   If `visualization` is set to `True`, the predictions of the automated model are visualized.

   .. code-block:: python

          if visualization:
              auto_model.plot_prediction()

8. **Return the Predictions**

   The function returns the predictions made by the automated model.

   .. code-block:: python

          return prediction

9. **Run the Example**

   The example is executed with a specified timeout and visualization.

   .. code-block:: python

      if __name__ == '__main__':
          set_random_seed(42)
          run_classification_example(timeout=10.0, visualization=True)

Usage
-----

To use this example, you can copy and paste the code into your Python environment. Ensure that the Fedot library is installed and that the paths to the datasets are correct. You can modify the `timeout`, `visualization`, and `with_tuning` parameters to suit your needs.

.. note::
   This example assumes that the required datasets are available at the specified paths within the Fedot project structure. If you are using different datasets, you will need to adjust the `train_data_path` and `test_data_path` variables accordingly.

.. note::
   For more information on the Fedot library and its capabilities, please refer to the `Fedot documentation <https://fedot.readthedocs.io/>`_.
