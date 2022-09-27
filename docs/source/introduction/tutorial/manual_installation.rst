How to setup the environment in the easiest way
-----------------------------------------------

The simplest way to install FEDOT is using ``pip``:

.. code-block::

  $ pip install fedot
  $ python -m venv <your_venv_path>

Installation with optional dependencies for image and text processing, and for DNNs:

.. code-block::

  $ pip install fedot[extra]
  $ python -m venv <your_venv_path>

Then, activate venv by using following command on Unix like OS

.. code-block::

  $ source venv/bin/activate
  # remember to call `deactivate` once you're done using the application

and the following command using Windows OS

.. code-block::

  $ venv\Scripts\activate
  # remember to call `venv\Scripts\deactivate` once you're done using the application

And that's it.