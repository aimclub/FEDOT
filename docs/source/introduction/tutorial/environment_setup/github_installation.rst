The GitHub way
--------------

-  **Step 1**. *Download FEDOT Framework*.

   -  First of all, you need to clone the FEDOT Framework to your personal computer.
      You can do it directly using the button `Code` (red square) and then copying
      a link to the repository (blue square).

         |Step 1|
   
   -  Then open cmd (Windows) or terminal (Unix-like OS), type in:

      .. code-block::
      
         $ git clone https://github.com/aimclub/FEDOT.git
         $ cd FEDOT

-  **Step 2**. *Creating VirtualEnv*.

   -  Next, you need to create a virtual environment in your project
      to avoid libraries incompatibility.
      To do this, type in
      
      .. code-block::

         $ python -m venv <your_venv_path>

      .. include:: ./activating_venv.rst

   -  After creating the virtual environment,
      install the libraries necessary for FEDOT to work.
      Type in:
      
      .. code-block:: 
      
         $ pip install .

   -  But, if you want to use additional functionality such as NNs,
      you'll need to run full installation option.

      To do this run the following command
   
      .. code-block::
      
         $ pip install .[extra]
      
      .. include:: ./extra_remark.rst

.. |Step 1| image:: github_download.png
