
.. _time_series_forecasting_example:

Time Series Forecasting Example
===========================================================

This example demonstrates the use of custom pipelines for time series forecasting, including optional tuning and visualization. The example uses a specific dataset and a predefined pipeline to make forecasts and evaluates the performance using mean squared error (MSE) and mean absolute error (MAE).

Overview
--------

The example is structured to perform the following tasks:

1. Load and prepare the dataset.
2. Apply a predefined pipeline to the dataset.
3. Evaluate the performance without tuning.
4. Optionally, tune the pipeline and re-evaluate the performance.
5. Visualize the results if required.

Step-by-Step Guide
------------------

1. **Import Necessary Libraries**

   The example starts by importing the necessary libraries and modules required for the forecasting task.

   .. code-block:: python

      import numpy as np
      from golem.core.tuning.simultaneous import SimultaneousTuner
      from sklearn.metrics import mean_squared_error, mean_absolute_error
      from examples.advanced.time_series_forecasting.composing_pipelines import visualise, get_border_line_info
      from examples.simple.time_series_forecasting.api_forecasting import get_ts_data
      from examples.simple.time_series_forecasting.ts_pipelines import ts_locf_ridge_pipeline
      from fedot.core.pipelines.pipeline import Pipeline
      from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
      from fedot.core.repository.metrics_repository import RegressionMetricsEnum

2. **Define the Experiment Function**

   The `run_experiment` function is defined to encapsulate the entire process of forecasting. It takes parameters for the dataset, pipeline, forecast length, and options for tuning and visualization.

   .. code-block:: python

      def run_experiment(dataset: str, pipeline: Pipeline, len_forecast=250, tuning=True, visualisalion=False):
          """ Example of ts forecasting using custom pipelines with optional tuning
          :param dataset: name of dataset
          :param pipeline: pipeline to use
          :param len_forecast: forecast length
          :param tuning: is tuning needed
          """
          # show initial pipeline
          pipeline.print_structure()

3. **Load and Prepare the Dataset**

   The dataset is loaded and split into training and testing sets. The target variable for the test set is also prepared.

   .. code-block:: python

          train_data, test_data, label = get_ts_data(dataset, len_forecast)
          test_target = np.ravel(test_data.target)

4. **Fit the Pipeline and Make Predictions**

   The pipeline is fitted on the training data and used to make predictions on the test data.

   .. code-block:: python

          pipeline.fit(train_data)
          prediction = pipeline.predict(test_data)
          predict = np.ravel(np.array(prediction.predict))

5. **Evaluate Performance Without Tuning**

   The performance of the pipeline without tuning is evaluated using RMSE and MAE.

   .. code-block:: python

          rmse = mean_squared_error(test_target, predict, squared=False)
          mae = mean_absolute_error(test_target, predict)
          metrics_info['Metrics without tuning'] = {'RMSE': round(rmse, 3),
                                                    'MAE': round(mae, 3)}

6. **Optionally Tune the Pipeline**

   If tuning is enabled, the pipeline is tuned using a tuner and the performance is re-evaluated.

   .. code-block:: python

          if tuning:
              tuner = TunerBuilder(train_data.task) \
                  .with_tuner(SimultaneousTuner) \
                  .with_metric(RegressionMetricsEnum.MSE) \
                  .with_iterations(300) \
                  .build(train_data)
              pipeline = tuner.tune(pipeline)
              pipeline.fit(train_data)
              prediction_after = pipeline.predict(test_data)
              predict_after = np.ravel(np.array(prediction_after.predict))

              rmse = mean_squared_error(test_target, predict_after, squared=False)
              mae = mean_absolute_error(test_target, predict_after)
              metrics_info['Metrics after tuning'] = {'RMSE': round(rmse, 3),
                                                      'MAE': round(mae, 3)}

7. **Visualize the Results**

   If visualization is enabled, the results are plotted.

   .. code-block:: python

          if visualisalion:
              visualise(plot_info)
              pipeline.print_structure()

8. **Run the Experiment**

   The experiment is run with specific parameters.

   .. code-block:: python

      if __name__ == '__main__':
          run_experiment('m4_monthly', ts_locf_ridge_pipeline(), len_forecast=10, tuning=True, visualisalion=True)

Usage
-----

To use this example, you can copy and paste the code into your Python environment. Ensure that you have the required libraries installed and that the dataset and pipeline are compatible with your use case. Adjust the parameters as needed to fit your specific forecasting task.

.. note::
   This example assumes that the necessary modules and datasets are available in the specified paths. Make sure to set up your environment accordingly.