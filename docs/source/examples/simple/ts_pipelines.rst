.. _ts_pipelines_doc:

Time Series Pipelines Documentation
===============================================================

This documentation provides an overview and detailed explanation of various time series analysis pipelines implemented using the `fedot.core.pipelines.pipeline_builder` module. Each pipeline is designed to handle different aspects of time series data, including preprocessing, feature engineering, and model training.

.. note::
    Ensure you have the necessary dependencies installed to run these pipelines.

.. _ts_ets_pipeline:

1. Exponential Smoothing Pipeline
---------------------------------

.. code-block:: python

    def ts_ets_pipeline():
        """
        Return pipeline with the following structure:

        .. image:: img_ts_pipelines/ts_ets_pipeline.png
          :width: 55%

        Where cut - cut part of dataset and ets - exponential smoothing
        """
        pip_builder = PipelineBuilder().add_node('cut').add_node('ets',
                                                             params={'error': 'add',
                                                                     'trend': 'add',
                                                                     'seasonal': 'add',
                                                                     'damped_trend': False,
                                                                     'seasonal_periods': 20})
        pipeline = pip_builder.build()
        return pipeline

This pipeline starts with a 'cut' operation to select a portion of the dataset, followed by an 'ets' (Exponential Smoothing) node for time series forecasting. The 'ets' node is configured with parameters specifying additive error, trend, and seasonal components.

.. _ts_ets_ridge_pipeline:

2. Exponential Smoothing with Ridge Pipeline
--------------------------------------------

.. code-block:: python

    def ts_ets_ridge_pipeline():
        """
        Return pipeline with the following structure:

        .. image:: img_ts_pipelines/ts_ets_ridge_pipeline.png
          :width: 55%

        Where cut - cut part of dataset, ets - exponential smoothing
        """
        pip_builder = PipelineBuilder() \
            .add_sequence(('cut', {'cut_part': 0.5}),
                          ('ets', {'error': 'add', 'trend': 'add', 'seasonal': 'add',
                                   'damped_trend': False, 'seasonal_periods': 20}),
                          branch_idx=0) \
            .add_sequence('lagged', 'ridge', branch_idx=1).join_branches('ridge')

        pipeline = pip_builder.build()
        return pipeline

This pipeline includes a sequence of operations starting with a 'cut' node to reduce the dataset size, followed by an 'ets' node. A separate branch with 'lagged' and 'ridge' nodes is then joined at the 'ridge' node.

.. _ts_glm_pipeline:

3. Generalized Linear Model Pipeline
------------------------------------

.. code-block:: python

    def ts_glm_pipeline():
        """
        Return pipeline with the following structure:

        .. image:: img_ts_pipelines/ts_glm_pipeline.png
          :width: 55%

        Where glm - Generalized linear model
        """
        pipeline = PipelineBuilder().add_node('glm', params={'family': 'gaussian'}).build()
        return pipeline

This simple pipeline uses a 'glm' (Generalized Linear Model) node with a Gaussian family for modeling.

.. _ts_glm_ridge_pipeline:

4. Generalized Linear Model with Ridge Pipeline
-----------------------------------------------

.. code-block:: python

    def ts_glm_ridge_pipeline():
        """
        Return pipeline with the following structure:

        .. image:: img_ts_pipelines/ts_glm_ridge_pipeline.png
          :width: 55%

        Where glm - Generalized linear model
        """
        pip_builder = PipelineBuilder() \
            .add_sequence('glm', branch_idx=0) \
            .add_sequence('lagged', 'ridge', branch_idx=1).join_branches('ridge')

        pipeline = pip_builder.build()
        return pipeline

This pipeline includes a 'glm' node in one branch and a sequence of 'lagged' and 'ridge' nodes in another, which are joined at the 'ridge' node.

.. _ts_polyfit_pipeline:

5. Polynomial Interpolation Pipeline
------------------------------------

.. code-block:: python

    def ts_polyfit_pipeline(degree):
        """
        Return pipeline with the following structure:

        .. image:: img_ts_pipelines/ts_polyfit_pipeline.png
          :width: 55%

        Where polyfit - Polynomial interpolation
        """
        pipeline = PipelineBuilder().add_node('polyfit', params={'degree': degree}).build()
        return pipeline

This pipeline uses a 'polyfit' node for polynomial interpolation, with the degree of the polynomial specified as a parameter.

.. _ts_polyfit_ridge_pipeline:

6. Polynomial Interpolation with Ridge Pipeline
-----------------------------------------------

.. code-block:: python

    def ts_polyfit_ridge_pipeline(degree):
        """
        Return pipeline with the following structure:

        .. image:: img_ts_pipelines/ts_polyfit_ridge_pipeline.png
          :width: 55%

        Where polyfit - Polynomial interpolation
        """
        pip_builder = PipelineBuilder() \
            .add_sequence(('polyfit', {'degree': degree}), branch_idx=0) \
            .add_sequence('lagged', 'ridge', branch_idx=1).join_branches('ridge')

        pipeline = pip_builder.build()
        return pipeline

This pipeline includes a 'polyfit' node in one branch and a sequence of 'lagged' and 'ridge' nodes in another, which are joined at the 'ridge' node.

.. _ts_complex_ridge_pipeline:

7. Complex Ridge Pipeline
-------------------------

.. code-block:: python

    def ts_complex_ridge_pipeline():
        """
        Return pipeline with the following structure:

        .. image:: img_ts_pipelines/ts_complex_ridge_pipeline.png
          :width: 55%

        """
        pip_builder = PipelineBuilder() \
            .add_sequence('lagged', 'ridge', branch_idx=0) \
            .add_sequence('lagged', 'ridge', branch_idx=1).join_branches('ridge')

        pipeline = pip_builder.build()
        return pipeline

This pipeline consists of two branches, each containing a 'lagged' and 'ridge' node, which are joined at the 'ridge' node.

.. _ts_complex_ridge_smoothing_pipeline:

8. Complex Ridge with Smoothing Pipeline
----------------------------------------

.. code-block:: python

    def ts_complex_ridge_smoothing_pipeline():
        """
        Pipeline looking like this

        .. image:: img_ts_pipelines/ts_complex_ridge_smoothing_pipeline.png
          :width: 55%

        Where smoothing - rolling mean
        """
        pip_builder = PipelineBuilder() \
            .add_sequence('smoothing', 'lagged', 'ridge', branch_idx=0) \
            .add_sequence('lagged', 'ridge', branch_idx=1).join_branches('ridge')

        pipeline = pip_builder.build()
        return pipeline

This pipeline includes a 'smoothing' node (rolling mean) followed by 'lagged' and 'ridge' nodes in one branch, and a 'lagged' and 'ridge' sequence in another, which are joined at the 'ridge' node.

.. _ts_complex_dtreg_pipeline:

9. Complex Decision Tree Regressor Pipeline
-------------------------------------------

.. code-block:: python

    def ts_complex_dtreg_pipeline(first_node='lagged'):
        """
        Return pipeline with the following structure:

        .. image:: img_ts_pipelines/ts_complex_dtreg_pipeline.png
          :width: 55%

        Where dtreg = tree regressor, rfr - random forest regressor
        """
        pip_builder = PipelineBuilder() \
            .add_sequence(first_node, 'dtreg', branch_idx=0) \
            .add_sequence(first_node, 'dtreg', branch_idx=1).join_branches('rfr')

        pipeline = pip_builder.build()
        return pipeline

This pipeline includes two branches, each starting with the specified 'first_node' followed by a 'dtreg' (Decision Tree Regressor) node, which are joined at the 'rfr' (Random Forest Regressor) node.

.. _ts_multiple_ets_pipeline:

10. Multiple Exponential Smoothing Pipeline
-------------------------------------------

.. code-block:: python

    def ts_multiple_ets_pipeline():
        """
        Return pipeline with the following structure:

        .. image:: img_ts_pipelines/ts_multiple_ets_pipeline.png
          :width: 55%

        Where ets - exponential_smoothing
        """
        pip_builder = PipelineBuilder() \
            .add_sequence('ets', branch_idx=0) \
            .add_sequence('ets', branch_idx=1) \
            .add_sequence('ets', branch_idx=2) \
            .join_branches('lasso')

        pipeline = pip_builder.build()
        return pipeline

This pipeline includes three 'ets' (Exponential Smoothing) nodes in separate branches, which are joined at the 'lasso' node.

.. _ts_ar_pipeline:

11. Auto Regression Pipeline
----------------------------

.. code-block:: python

    def ts_ar_pipeline():
        """
        Return pipeline with the following structure:

        .. image:: img_ts_pipelines/ts_ar_pipeline.png
          :width: 55%

        Where ar - auto regression
        """
        pipeline = PipelineBuilder().add_node('ar').build()
        return pipeline

This simple pipeline uses an 'ar' (Auto Regression) node for time series forecasting.

.. _ts_arima_pipeline:

12. ARIMA Pipeline
------------------

.. code-block:: python

    def ts_arima_pipeline():
        """
        Return pipeline with the following structure:

        .. image:: img_ts_pipelines/ts_arima_pipeline.png
          :width: 55%

        """
        pipeline = PipelineBuilder().add_node("arima").build()
        return pipeline

This pipeline uses an 'arima' node for time series forecasting, implementing the AutoRegressive Integrated Moving Average model.

.. _ts_stl_arima_pipeline:

13. STL-ARIMA Pipeline
----------------------

.. code-block:: python

    def ts_stl_arima_pipeline():
        """
        Return pipeline with the following structure:

        .. image:: img_ts_pipelines/ts_stl_arima_pipeline.png
          :width: 55%

        """
        pipeline = PipelineBuilder().add_node("stl_arima").build()
        return pipeline

This pipeline uses an 'stl_arima' node, which combines Seasonal and Trend decomposition using Loess with the ARIMA model for time series forecasting.

.. _ts_locf_ridge_pipeline:

14. LOCF Ridge Pipeline
-----------------------

.. code-block:: python

    def ts_locf_ridge_pipeline():
        """
        Pipeline with naive LOCF (last observation carried forward) model
        and lagged features

        .. image:: img_ts_pipelines/ts_locf_ridge_pipeline.png
          :width: 55%

        """
        pip_builder = PipelineBuilder() \
            .add_sequence('locf', branch_idx=0) \
            .add_sequence('ar', branch_idx=1) \
            .join_branches('ridge')

        pipeline = pip_builder.build()
        return pipeline

This pipeline includes a 'locf' node for handling missing values using the Last Observation Carried Forward method, followed by an 'ar' node for auto regression, which is then joined with a 'ridge' node.

.. _ts_naive_average_ridge_pipeline:

15. Naive Average Ridge Pipeline
--------------------------------

.. code-block:: python

    def ts_naive_average_ridge_pipeline():
        """
        Pipeline with simple forecasting model (the forecast is mean value for known
        part)

        .. image:: img_ts_pipelines/ts_naive_average_ridge_pipeline.png
          :width: 55%

        """
        pip_builder = PipelineBuilder() \
            .add_sequence('ts_naive_average', branch_idx=0) \
            .add_sequence('lagged', branch_idx=1) \
            .join_branches('ridge')

        pipeline = pip_builder.build()
        return pipeline

This pipeline starts with a 'ts_naive_average' node for simple forecasting based on the mean value of known data, followed by a 'lagged' node, which is then joined with a 'ridge' node.

.. _cgru_pipeline:

16. Convolutional GRU Pipeline
------------------------------

.. code-block:: python

    def cgru_pipeline(window_size=200):
        """
        Return pipeline with the following structure:

        .. image:: img_ts_pipelines/cgru_pipeline.png
          :width: 55%

        Where cgru - convolutional long short-term memory model
        """
        pip_builder = PipelineBuilder() \
            .add_sequence('lagged', 'ridge', branch_idx=0) \
            .add_sequence(('lagged', {'window_size': window_size}), 'cgru', branch_idx=1) \
            .join_branches('ridge')

        pipeline = pip_builder.build()
        return pipeline

This pipeline includes a 'lagged' node with a specified window size followed by a 'cgru' (Convolutional GRU) node in one branch, and a 'lagged' and 'ridge' sequence in another, which are joined at the 'ridge' node.

This documentation provides a comprehensive guide to the various time series analysis pipelines available, each tailored to specific needs and scenarios. Users can copy and adapt these pipelines for their own projects, ensuring they understand the underlying logic and configuration of each node.