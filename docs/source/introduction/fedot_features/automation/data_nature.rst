Data nature
-----------

FEDOT uses specific data processing according to the source
of the input data (whether it is pandas DataFrame, or numpy array, or just a path to the dataset, etc).

.. note::

    Be careful with datetime features, as they are to be casted into a float type with milliseconds unit.


Apart from that FEDOT is capable of working with multi-modal data.
It means that you can pass it different types of datasets
(tables, texts, images, etc) and it will get the information from within to work with it.

.. seealso::
    `Multimodal data classification jupyter-notebook example <https://github.com/ITMO-NSS-team/fedot-examples/blob/main/notebooks/latest/7_multimodal_data.ipynb>`_