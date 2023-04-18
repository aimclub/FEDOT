AutoML capabilities
===================

FEDOT is capable of setting its 'automation rate' by omitting some of its parameters.
For example, if you just create the FEDOT instance and call the ``fit`` method with the appropriate dataset on it,
you will have a full automation of the learning process,
see :doc:`automated composing </introduction/tutorial/composing_pipelines/automated_creation>`

At the same time, if you pass some of the parameters, you will have a partial automation,
see :doc:`manual composing </introduction/tutorial/composing_pipelines/manual_creation>`.

Other than that, you can utilize more AutoML means:

.. toctree::
   :glob:
   :maxdepth: 1

   data_nature
   dimensional_operations
   models_used
