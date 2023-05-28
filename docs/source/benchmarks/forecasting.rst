Time series forecasting
-----------------------

With FEDOT it is possible to effectively forecast time series. In our research papers, we make detailed comparisons on various datasets with other libraries. Below are some results of such comparisons.



We used subsample from `M4 competition <https://paperswithcode.com/dataset/m4>`__ (subsample contains 471 series with daily, weekly, monthly, quarterly, yearly intervals). Horizons for forecasting were six for yearly, eight for quarterly, 18 for monthly series, 13 for weekly series and 14 for daily. The metric for estimation is Symmetric Mean Absolute Percentage Error (SMAPE).

The results of comparison with competing libraries averaged for all time series in each dataset by SMAPE. The errors are provided for different forecast horizons and shown by quantiles (q) as 10th, 50th (median) and 90th. The smallest error values on the quantile are shown in bold.

.. csv-table:: Classification statistics
   :file: amlb_res.csv
   :align: center
   :widths: auto
   :header-rows: 1
