Data preprocessing
------------------

FEDOT uses two types of preprocessing: obligatory and optional.

.. note::

    Preprocessing is optional (see ``use_input_preprocessing`` :doc:`main API parameter </api/api>`),
    so you can save some time if you have already preprocessed dataset.

**The first one**, as you might guess, cares about something important that can break or just complicate 
your program, these are:

.. list-table:: Obligatory preprocessing
   :widths: 25 5
   :header-rows: 1

   * - Problem
     - Solution
   * - 'inf' values in features
     - replace
   * - huge amount of nans in features or targets
     - drop
   * - binary categorical form of the features or targets
     - binarize
   * - extra spaces in categorical features
     - trim

**The second one** depends on composed pipeline structure, and is applied only if
it is necessary for the next model from a processing queue to work.

.. list-table:: Optional preprocessing
   :widths: 10 5
   :header-rows: 1

   * - Problem
     - Solution
   * - nans in features
     - impute
   * - non-binary categorical features
     - LabelEncode or OneHotEncode

But depending on the pipeline structure, it might be ommited:

.. image:: ../img_intro/optional_preprocessing_condition.png
   :width: 100%
   :alt: Optional preprocessing condition

.. note::

    Both of the technics applied only once.