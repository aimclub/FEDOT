Data Preprocessing
========

Preprocessing for tabular data
--------

Preprocessing for tabular data in FEDOT can be represented as the following block diagram:

|Block diagram|

Such approach to preprocessing allows to get the real data type
and minimize the number of dropped columns due to unrecognized data.


Also for more flexible approach to preprocessing there are 2 variables to control data conversion:

- numerical_min_uniques -- if number of unique values in the column lower, than threshold - convert column into categorical. Default: 13
- categorical_max_classes_th -- if categorical column contains too much unique values convert it into numerical. Default: None



.. |Block diagram| image:: img_utilities/preprocessing_tabular.png
