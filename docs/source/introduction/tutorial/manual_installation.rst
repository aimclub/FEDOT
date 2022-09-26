How to setup the environment manually
-------------------------------------

The easiest way to install FEDOT is using ``pip``:

.. code-block::

  $ pip install fedot

Alternatively, in order to work with the source code:

.. code-block::

   $ git clone https://github.com/nccr-itmo/FEDOT.git
   $ cd FEDOT
   $ pip install .
   $ pytest -s test

Installation with optional dependencies for image and text processing, and for DNNs:

.. code-block::

  $ pip install fedot[extra]

Or by dealing with the source code:

.. code-block::

   $ git clone https://github.com/nccr-itmo/FEDOT.git
   $ cd FEDOT
   $ pip install .[extra]
   $ pytest -s test
