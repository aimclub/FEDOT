API Usage
==========

Why *feature_name* is not supported?
------------------------------------

    *We provide a constant extension of Fedot’s feature set. However, any
    Pull Requests and issues from external contributors that introduce or
    suggests the new features will be appreciated. You can create your* `pull
    request`_ *or* `issue`_ *in the main repository of Fedot.*


.. List of links:

.. _pull request: https://github.com/nccr-itmo/FEDOT/pulls
.. `pull request` replace:: *pull request*

.. _issue: https://github.com/nccr-itmo/FEDOT/issues
.. `issue` replace:: *issue*

Can I change path to cacher's database files?
---------------------------------------------

    *Using* :doc:`FEDOT’s main API </api/api>` *you can change*
    ``cache_dir`` *parameter with your custom path to
    the database files.*
    
    *So, setting it would look like:*
    
    .. code:: python

       your_custom_path = ...
       ...
       model = Fedot(..., cache_dir=your_custom_path, ...)
       ...