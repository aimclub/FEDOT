Data Preprocessing
========


Main ideas about preprocessing
--------

There are two steps of preprocessing in FEDOT: on API and fit levels:

- Preprocessing on API level
    On this level FEDOT determines what type of data was passed to the input and selects a strategy for
    reading data in according to the type.

    Then recommendations are applied to the data. Recommendations applies only in safe_mode
    so if safe_mode is True than FEDOT will cut large datasets to prevent memory overflow
    and use label encoder instead of OneHotEncoder if amount of unique values of categorical features is high.

- Preprocessing on fit level
    On this level there are two more levels of preprocessing: obligatory and optional.
    On *obligatory level* incorrect values are removed, extra spaces are cleaned, the types of values in the data are determined and,
    if there are several of them, then the data is converted to a single type.
    Also at this level, data columns are deleted if they could not be cast to the same type.

    On *optional level* gaps are filled in and categorical encoding applied if needed.


General scheme of preprocessing
--------

Preprocessing for tabular data in FEDOT can be represented as the following block diagram:

|Block diagram|

Examples of preprocessing
--------------

Such approach to preprocessing allows to get the real data type
and minimize the number of dropped columns due to unrecognized data

The processing of the following samples of data well demonstrates the result of preprocessing in FEDOT.

- gap filling:

|gap filling|

-column remove if too many nans:

|nans|

- column revome if the data is too ambiguous:

|failed ratio|

- cast to a single type:

|one type|

- reduction to a binary classification problem:

|binary|


Additional features
---------

Also for more flexible approach to preprocessing there are 2 variables to control data conversion:

- numerical_min_uniques -- if number of unique values in the column lower, than threshold - convert column into categorical. Default: 13
- categorical_max_classes_th -- if categorical column contains too much unique values convert it into numerical. Default: None

For example, for column converting to numerical if the number of unique values in the column is greater than 5:

.. code:: python

    # pipeline for which to set params
    pipeline = Pipeline(PrimaryNode('dt'))
    pipeline = correct_preprocessing_params(pipeline, numerical_min_uniques=5)

After this preprocessing with this pipeline will be performed according to the specified conditions.


.. |gap filling| image:: img_utilities/gap_filling.jpg
   :width: 25%

.. |nans| image:: img_utilities/nans.jpg
   :width: 25%

.. |failed ratio| image:: img_utilities/failed_ratio.jpg
   :width: 25%

.. |one type| image:: img_utilities/cast_to_one_type.jpg
   :width: 25%

.. |binary| image:: img_utilities/binary.jpg
   :width: 25%

.. |Block diagram| image:: img_utilities/fedot_preprocessing_tabular.png
   :width: 70%
