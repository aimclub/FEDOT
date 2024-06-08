.. _fitted_time_series_example:

=========================================================================
Example: Visualizing Fitted Time Series Values
=========================================================================

This example demonstrates how to use the FEDOT framework to obtain and visualize fitted values of a time series. Fitted values are the predictions made by a model on the training data, which help in understanding how well the model captures the underlying structure of the time series.

.. code-block:: python

    from matplotlib import pyplot as plt
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.ts_wrappers import fitted_values, in_sample_fitted_values
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
    from test.unit.pipelines.test_pipeline_ts_wrappers import get_simple_short_lagged_pipeline

    def show_fitted_time_series(len_forecast=24):
        """
        Shows an example of how to get fitted values of a time series by any
        pipeline created by FEDOT

        fitted values - are the predictions of the pipelines on the training sample.
        For time series, these values show how well the model reproduces the time
        series structure
        """
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=len_forecast))

        ts_input = InputData.from_csv_time_series(file_path='../../../cases/data/time_series/metocean.csv',
                                                  task=task, target_column='value')

        pipeline = get_simple_short_lagged_pipeline()
        train_predicted = pipeline.fit(ts_input)

        # Get fitted values for every 10th forecast
        fitted_ts_10 = fitted_values(ts_input, train_predicted, 10)
        # Average for all forecasting horizons
        fitted_ts_act = fitted_values(ts_input, train_predicted)
        # In-sample forecasting fitted values
        in_sample_validated = in_sample_fitted_values(ts_input, train_predicted)

        plt.plot(range(len(ts_input.idx)), ts_input.target, label='Actual time series', alpha=0.8)
        plt.plot(fitted_ts_10.idx, fitted_ts_10.predict, label='Fitted values horizon 10', alpha=0.2)
        plt.plot(fitted_ts_act.idx, fitted_ts_act.predict, label='Fitted values all', alpha=0.2)
        plt.plot(in_sample_validated.idx, in_sample_validated.predict, label='In-sample fitted values')
        plt.legend()
        plt.grid()
        plt.show()

    if __name__ == '__main__':
        show_fitted_time_series()

Step-by-Step Guide
------------------

1. **Task Definition**:
   The task is defined as time series forecasting with a specified forecast length.

   .. code-block:: python

       task = Task(TaskTypesEnum.ts_forecasting,
                   TsForecastingParams(forecast_length=len_forecast))

2. **Data Loading**:
   The time series data is loaded from a CSV file.

   .. code-block:: python

       ts_input = InputData.from_csv_time_series(file_path='../../../cases/data/time_series/metocean.csv',
                                                 task=task, target_column='value')

3. **Pipeline Creation and Training**:
   A pipeline is created and trained on the input data.

   .. code-block:: python

       pipeline = get_simple_short_lagged_pipeline()
       train_predicted = pipeline.fit(ts_input)

4. **Obtaining Fitted Values**:
   Fitted values are calculated for different scenarios (every 10th forecast, all forecasts, and in-sample).

   .. code-block:: python

       fitted_ts_10 = fitted_values(ts_input, train_predicted, 10)
       fitted_ts_act = fitted_values(ts_input, train_predicted)
       in_sample_validated = in_sample_fitted_values(ts_input, train_predicted)

5. **Visualization**:
   The actual time series and the fitted values are plotted.

   .. code-block:: python

       plt.plot(range(len(ts_input.idx)), ts_input.target, label='Actual time series', alpha=0.8)
       plt.plot(fitted_ts_10.idx, fitted_ts_10.predict, label='Fitted values horizon 10', alpha=0.2)
       plt.plot(fitted_ts_act.idx, fitted_ts_act.predict, label='Fitted values all', alpha=0.2)
       plt.plot(in_sample_validated.idx, in_sample_validated.predict, label='In-sample fitted values')
       plt.legend()
       plt.grid()
       plt.show()

This example provides a clear demonstration of how to use FEDOT to obtain and visualize fitted values of a time series, which is crucial for assessing the model's performance on the training data.