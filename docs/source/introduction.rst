FEDOT
=====

FEDOT - an open-source framework for automated modeling and machine learning (AutoML). It can build custom modeling pipelines for different real-world processes in an automated way using an evolutionary approach. FEDOT supports classification (binary and multiclass), regression, clustering, and time series prediction tasks.

The framework is not limited to specific AutoML tasks (such as pre-processing of input data, feature selection, or optimization of model hyperparameters), but allows you to solve a more general structural learning problem - for a given data set, a solution is built in the form of a graph (DAG), the nodes of which are represented by ML models, pre-processing procedures, and data transformation.


Main features
=============

The main features of the framework are as follows:

- The FEDOT architecture is highly flexible and therefore the framework can be used to automate the creation of mathematical models for various problems, types of data, and models;
- FEDOT already supports popular ML libraries (scikit-learn, keras, statsmodels, etc.), but you can also integrate custom tools into the framework if necessary;
- Pipeline optimization algorithms are not tied to specific data types or tasks, but you can use special templates for a specific task class or data type (time series forecasting, NLP, tabular data, etc.) to increase the efficiency;
- The framework is not limited only to machine learning, it is possible to embed models related to specific areas into pipelines (for example, models in ODE or PDE);
- Additional methods for hyperparameters tuning can be seamlessly integrated into FEDOT (in addition to those already supported);
- The resulting pipelines can be exported in a human-readable JSON format, which allows you to achieve reproducibility of the experiments.

Thus, compared to other frameworks, FEDOT:

- Is not limited to specific modeling tasks and claims versatility and expandability;
- Supports the the complex modelling pipelines with variable shape and structure;
- Allows building models using input data of various nature (texts, images, tables, etc.) and consisting of different types of models.
- Allows managing the complexity of models and thereby achieving better results.


Quick start
===========

FEDOT Framework quick start guide

How to install
--------------
.. code::

 pip install fedot

How to create your own composite model in manual way
----------------------------------------------------

-  **Step 1**. Specify problem type and choose dataset.

.. code:: python

   import pandas as pd

   model = Fedot(problem='classification')

   dataset_to_train = pd.read_csv(train_file_path)
   dataset_to_validate = pd.read_csv(train_file_path)

-  **Step 2**. Create *Pipeline* instance, create nodes with desired models

.. code:: python

   node_first = PrimaryNode('logit')
   node_second= PrimaryNode('xgboost')
   node_final = SecondaryNode('knn', nodes_from = [node_first, node_second])
   pipeline = Pipeline(node_final)

-  **Step 3**. Fit the pipeline using *fit* method.

.. code:: python

   model.fit(features=dataset_to_train, target='target', predefined_model=pipeline)

-  **Step 4**. Obtain the prediction using *predict* method.

.. code:: python

   prediction = model.predict(features=dataset_to_validate)

How to compose the pipeline in automated way
-----------------------------------------

.. code:: python

   auto_model = Fedot(problem='classification')
   pipeline = auto_model.fit(features=dataset_to_train, target='target')
   prediction = auto_model.predict(features=dataset_to_validate)
   auto_metrics = auto_model.get_metrics()

How to setup the development environments for the Fedot
-------------------------------------------------------

-  **Step 1**. *Download FEDOT Framework*.

   -  First of all, you need to clone the FEDOT Framework to your personal computer. You can do it directly using the button 'clone or download' (red square) or you can install IDE (e.g. PyCharm) and using the 'clone in Pycharm' button (blue square), which will open the files you need directly in the Pycharm project.

   -  For more details, take a look at the picture below.

        |Step 1|

-  **Step 2**. *Creating VirtualEnv in Pycharm project*.

   -  Next, you need to create virtual enviroment in your Pycharm
      project. To do this, go through the following sections:
      'File - Settings - Project Interpreter - Add new'.
   -  For more details, take a look at the picture below.

        |Step 2|

   -  After you have created a virtual environment, you should install
      the libraries necessary for the FEDOT framework to work. In order
      to do this, go to the terminal console (blue square) and run the
      following command *pip install .[extra]* (red square).
   -  For more details, take a look at the picture below.

        |Step 3|

-  **Step 3**. *Manually installing libraries*.

   -  In order to use the

.. |Step 1| image:: img/img_tutorial/1_step.png
.. |Step 2| image:: img/img_tutorial/2_step.png
.. |Step 3| image:: img/img_tutorial/3_step.png
