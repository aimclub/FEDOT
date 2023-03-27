Data preprocessing
------------------

FEDOT uses two types of preprocessing: obligatory and optional.

.. note::

    Preprocessing is optional (see ``use_input_preprocessing`` :doc:`main API parameter </api/api>`),
    so you can save some time if you have already preprocessed dataset.

**The first one**, as you might guess, cares about something important that can break or just complicate 
your program, these are:

* 'inf' values in features
* huge amount of nans in features or targets
* categorical form of the binary columns or targets
* extra spaces in categorical columns.

**The second one** depends on composed pipeline structure, and is applied only if
it is necessary for the next model from a processing queue to work.

.. note::

    Both of the technics applied only once.