
.. _cgru_forecasting_example:

=========================================================================
Example: CGru Forecasting with Fedot Pipeline
=========================================================================

Overview
--------

This example demonstrates the use of the Fedot framework to build and apply a forecasting pipeline using the CGru model for time series data. The pipeline is designed to predict future values of a time series based on historical data. The example includes data preparation, model fitting, prediction, and visualization of the results.

Step-by-Step Guide
------------------

1. Importing Necessary Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    from examples.advanced.time_series_forecasting.composing_pipelines import visualise, get_border_line_info
    from examples.simple.time_series_forecasting.api_forecasting import get_ts_data
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder

This block imports the required libraries and functions for the example. It includes NumPy for numerical operations, Scikit-learn for calculating error metrics, and specific functions and classes from the Fedot framework for time series forecasting and pipeline construction.

2. Defining the Forecasting Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def cgru_forecasting():
        """ Example of cgru pipeline serialization """
        horizon = 12
        window_size = 200
        train_data, test_data = get_ts_data('salaries', horizon)

This function initializes the forecasting process. It sets the forecasting horizon and window size, and retrieves the training and testing data for the 'salaries' time series.

3. Building the Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

        pipeline = PipelineBuilder().add_node("lagged", params={'window_size': window_size}).add_node("cgru").build()

The pipeline is constructed using the PipelineBuilder. It includes a 'lagged' preprocessing node with a specified window size and a 'cgru' model node for the actual forecasting.

4. Fitting the Model and Making Predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

        pipeline.fit(train_data)
        prediction = pipeline.predict(test_data).predict[0]

The pipeline is fitted on the training data, and predictions are made on the test data.

5. Preparing Data for Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

        plot_info = [
            {'idx': np.concatenate([train_data.idx, test_data.idx]),
             'series': np.concatenate([test_data.features, test_data.target]),
             'label': 'Actual time series'},
            {'idx': test_data.idx,
             'series': np.ravel(prediction),
             'label': 'prediction'},
            get_border_line_info(test_data.idx[0],
                                 prediction,
                                 np.ravel(np.concatenate([test_data.features, test_data.target])),
                                 'Border line')
        ]

Data is prepared for visualization, including the actual time series, predictions, and a border line.

6. Calculating Error Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

        rmse = mean_squared_error(test_data.target, prediction, squared=False)
        mae = mean_absolute_error(test_data.target, prediction)
        print(f'RMSE - {rmse:.4f}')
        print(f'MAE - {mae:.4f}')

Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) are calculated and printed.

7. Visualizing the Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

        visualise(plot_info)

The results are visualized using the `visualise` function.

8. Running the Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    if __name__ == '__main__':
        cgru_forecasting()

The example is executed if the script is run as the main program.

Conclusion
----------

This example provides a comprehensive guide on how to use the Fedot framework to create a forecasting pipeline with the CGru model. It covers data retrieval, pipeline construction, model fitting, prediction, error calculation, and visualization. Users can adapt this example to their own time series forecasting tasks by modifying the data source and pipeline configuration.