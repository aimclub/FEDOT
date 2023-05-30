Time series forecasting
-----------------------


With FEDOT it is possible to effectively forecast time series. In our research papers, we make detailed comparisons on various datasets with other libraries. Below there are some results of such comparisons.



Here we used subsample from `M4 competition <https://paperswithcode.com/dataset/m4>`__ (subsample contains 461 series with daily, weekly, monthly, quarterly, yearly intervals). Horizons for forecasting were six for yearly, eight for quarterly, 18 for monthly series, 13 for weekly series and 14 for daily. The metric for estimation is Symmetric Mean Absolute Percentage Error (SMAPE).

The results of comparison with competing libraries averaged for all time series in each interval by SMAPE (%). The errors are provided for different forecast horizons and shown by quantiles (q) as 10th, 50th (median) and 90th. The smallest error values on the quantile are shown in bold.
Timeout for Fedot and other frameworks was set by 2 minutes on each series. For TPOT and H2O (which do not support forecasting natively) lagged transformation was used.

    +----------+----------+-----------+---------+---------+-----------+---------+---------+
    | Library  | Quantile |                   Intervals                                   |
    +          +          +-----------+---------+---------+-----------+---------+---------+
    |          |          |   Daily   | Weekly  | Montly  | Quarterly | Yearly  |  Overall|
    +==========+==========+===========+=========+=========+===========+=========+=========+
    |  AutoTS  |    10    |   **0,79**|  0,85   |  0,81   | **1,66**  |**1,84** |1,03     |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    50    |   2,37    |  5,29   |  5,88   |    7,1    |   9,25  | 5,14    |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    90    |   7,29    | 25,31   |**34,73**|   43,54   |  40,41  |30,11    |
    +----------+----------+-----------+---------+---------+-----------+---------+---------+
    |   TPOT   |    10    |    1,2    |  1,62   |  1,49   |    2,4    |  3,01   |1,48     |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    50    |   2,28    |  6,21   |  6,58   |   9,12    | **7,72**|5,49     |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    90    | **6,71**  |  20,3   | 39,14   |   53,79   | 70,71   |30,53    |
    +----------+----------+-----------+---------+---------+-----------+---------+---------+
    |   H2O    |    10    |   1,14    |  1,32   |  1,34   |   3,44    |  4,05   |1,44     |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    50    |   2,28    |  6,75   |  7,87   |   10,1    | 15,9    |6,76     |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    90    |   8,23    | 22,59   | 41,05   |   39,35   |  63,02  |29,78    |
    +----------+----------+-----------+---------+---------+-----------+---------+---------+
    | pmdarima |    10    |   0,89    |  1,48   |  2,06   |   2,28    |  7,67   |1,5      |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    50    |   2,33    |  7,47   |  7,45   |   9,91    | 16,97   |6,82     |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    90    |   8,13    | 33,23   | 47,04   |   40,97   | 67,32   |38,96    |
    +----------+----------+-----------+---------+---------+-----------+---------+---------+
    |Autogluon |    10    |   0,98    |0,85     | **0,76**|   2       |  2,72   |  1,02   |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    50    |   2,3     |5,26     |**4,9**  | **6,97**  |  9,53   |**4,55** |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    90    |   7,41    |22,08    |**33,83**| **27,48** | 44,66   |26,78    |
    +----------+----------+-----------+---------+---------+-----------+---------+---------+
    |  Fedot   |    10    |   0,92    |**0,73** |  1,25   |   1,98    |  2,18   |**1,01** |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    50    | **2,04**  |**4,06** |  5,53   |   8,42    |  9,51   |  4,66   |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    90    |   6,78    |**18,06**|  34,73  |   34,26   |**37,39**|**26,01**|
    +----------+----------+-----------+---------+---------+-----------+---------+---------+

Additionally you can examine papers about Fedot performance on different time series forecasting tasks `[1] <https://link.springer.com/chapter/10.1007/978-3-031-16474-3_45>`__ , `[2] <https://arpgweb.com/journal/7/special_issue/12-2018/5/&page=6>`__, `[3] <https://ieeexplore.ieee.org/document/9870347>`__,
`[4] <https://ieeexplore.ieee.org/document/9870347>`__,  `[5] <https://ieeexplore.ieee.org/document/9870347>`__,  `[6] <https://www.mdpi.com/2073-4441/13/24/3482/htm>`__,  `[7] <https://ieeexplore.ieee.org/abstract/document/9986887>`__.
