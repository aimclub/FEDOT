FEDOT
============

.. start-badges
.. list-table::
    :stub-columns: 1

    * - package
      - | |pypi| |py_6| |py_7| |py_8|
    * - tests
      - | |build| |coverage|
    * - docs
      - |docs|
    * - license
      - | |license|
    * - downloads
      -  .. image:: https://static.pepy.tech/personalized-badge/fedot?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
            :target: https://pepy.tech/project/fedot
    * - support
      - .. image:: https://img.shields.io/badge/Telegram-Group-blue.svg
           :target: https://t.me/FEDOT_helpdesk
           :alt: Telegram Chat


.. end-badges

Introduction
==============

This repository contains FEDOT - a open source framework for automated modeling and machine learning (AutoML). It can build custom modelling pipelines for the different real-world processes in an automated way using an evolutionary approach. FEDOT supports classification (binary and multiclass), regression, clustering, and time series prediction tasks.

The main feature in the framework is complex management of interactions between various computing elements of pipelines. First of all, this includes the stage of machine learning model design. FEDOT allows you to not just choose the best version of the model and train it, but to create a complex (composite) model. It allows you to share several models of different complexity, which helps you to achieve better modeling quality than when using any of these models separately. Within the framework, we describe composite models in the form of a graph defining the connections between data preprocessing blocks and model blocks.

The framework is not limited to individual AutoML tasks, such as pre-processing of initial data, selection of features or optimization of model hyperparameters, but allows you to solve a more general structural learning problem - for a given data set, a solution is built in the form of a graph (DAG), the nodes of which are represented by ML-models, pre-processing procedures and data transformation.

The intro video about Fedot is available here:


.. image:: https://res.cloudinary.com/marcomontalbano/image/upload/v1606396758/video_to_markdown/images/youtube--RjbuV6i6de4-c05b58ac6eb4c4700831b2b3070cd403.jpg
    :target: http://www.youtube.com/watch?v=RjbuV6i6de4
    :alt: Introducing Fedot

The project is maintained by the research team of Natural Systems Simulation Lab, which is a part of the National Center for Cognitive Research of ITMO University.


FEDOT features
==============

The main features of the framework are as follows:

 * The FEDOT architecture has high flexibility and therefore the framework can be used to automate the creation of mathematical models for various problems, different types of data and models;
 * FEDOT supports popular ML libraries (scikit-learn, keras, statsmodels, etc.), but if necessary, you can integrate other tools into it;
 * Pipeline optimization algorithms are not tied to data types or tasks, but to increase efficiency, you can use special templates for a specific task class or data type - time series prediction, NLP, tabular data, etc.;
 * The framework is not limited only to machine learning, it is possible to embed models related to specific areas into pipelines (for example, models in ODE or PDE);
 * Additional methods for tuning hyperparameters of different models can also be "seamlessly" added to FEDOT (in addition to those already supported);
 * FEDOT supports any-time mode of operation: at any time, you can stop the algorithm and get the result;
 * The resulting pipelines can be exported in a convenient json format without binding to the framework, which allows you to achieve reproducibility of the experiment.

Thus, compared to other frameworks, FEDOT:

 * Is not limited to one task type, but claims versatility and expandability;
 * Allows to more flexibly manage the complexity of models and thereby achieve better results.
 * Allows build models using input data of various nature - texts, images, tables, etc., and consisting of different types of models.

Installation
============

Common installation:

.. code-block::

   $ pip install fedot

In order to work with FEDOT source code:

.. code-block::

    $ git clone https://github.com/nccr-itmo/FEDOT.git
    $ cd FEDOT
    $ pip install -r requirements.txt
    $ pytest -s test


How to use (simple approach)
==========

FEDOT provides a high-level API that allows you to use its capabilities simpler.
At the moment, API can be used for classification and regression tasks only.
But the time series forecasting and clustering support will be implemented soon (you still can solve these tasks via advanced initialization, see above).
Input data must be ether in numpy-array format or CSV files.

To use API, follow these steps:

1. Import Fedot class

.. code-block:: python

  from fedot.api.api_runner import Fedot

2. Select the type of modelling problem and the hyperparameters of optimisation algorithm (optional).

.. code-block:: python

    task = 'classification'
    composer_params = {'max_depth': 2,
                       'learning_time': 10}

3. Initialize Fedot object with parameters. It provides a ML-popular fit/predict interface:

- fedot.fit runs optimization and returns the resulted composite model
- fedot.predict returns the predictied values for a given features
- fedot.quality_metric calculates the quality metrics of predictions

.. code-block:: python

  train_file = pd.read_csv(train_file_path)
  x, y = train_file.loc[:, ~train_file.columns.isin(['target'])].values, train_file['target'].values
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=24)

  model = Fedot(ml_task=task,
                composer_params=composer_params)
  fedot_model = model.fit(features=x_train,
                          target=y_train)
  prediction = model.predict(features=x_test)
  metric = model.quality_metric(target=y_test)


How to use (advanced approach)
==========

The main purpose of FEDOT is to identify a suitable composite model for a given dataset.
The model is obtained via optimization process (we also call it 'composing') that can be fine-tuned if necessary.\
Firstly, you need to prepare datasets for fit and validate and specify a task that you going to solve:

.. code-block:: python

  task = Task(TaskTypesEnum.classification)
  dataset_to_compose = InputData.from_csv(train_file_path, task=task)
  dataset_to_validate = InputData.from_csv(test_file_path, task=task)

Then, chose a set of models that can be included in the composite model, and the optimized metric function:

.. code-block:: python

  available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)
  metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

Next, you need to specify requirements for composer.
In this case, GPComposer is chosen that is based on evolutionary algorithm.

.. code-block:: python

  composer_requirements = GPComposerRequirements(
    primary=available_model_types,
    secondary=available_model_types, max_arity=3,
    max_depth=3, pop_size=20, num_of_generations=20,
    crossover_prob=0.8, mutation_prob=0.8, max_lead_time=20)

After that you need to initialize composer with builder using specified parameters:

.. code-block:: python

 builder = GPComposerBuilder(task=task).with_requirements(composer_requirements) \
        .with_metrics(metric_function) \
        .with_optimiser_parameters(optimiser_parameters)
 composer = builder.build()

Now you can run the optimization and obtain a composite model:

.. code-block:: python

  chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                              initial_chain=None,
                                              composer_requirements=composer_requirements,
                                              metrics=metric_function,
                                              is_visualise=False)

Finally, you can test the resulted model on the validation dataset:

.. code-block:: python

  roc_on_valid_evo_composed = calculate_validation_metric(chain_evo_composed,
                                                          dataset_to_validate)
  print(f'Composed ROC AUC is {roc_on_valid_evo_composed:.3f}')


Examples & Tutorials
====================

Jupyter notebooks with tutorials are located in the "notebooks" folder. There you can find the following guides:

* `Time series forecasting tutorial <https://github.com/nccr-itmo/FEDOT/tree/master/notebooks/time_series_forecasting/Time%20series%20forecasting%20with%20FEDOT.ipynb>`__

Extended examples:

- Credit scoring problem, i.e. `binary classification task <https://github.com/nccr-itmo/FEDOT/blob/master/cases/credit_scoring_problem.py>`__
- Time series forecasting, i.e. `random process regression <https://github.com/nccr-itmo/FEDOT/blob/master/cases/metocean_forecasting_problem.py>`__
- Spam detection, i.e. `natural language preprocessing <https://github.com/nccr-itmo/FEDOT/blob/master/cases/spam_detection.py>`__


Also, several video tutorials are `available <https://www.youtube.com/playlist?list=PLlbcHj5ytaFUjAxpZf7FbEaanmqpDYhnc>`__ (in Russian).

Project structure
=================

The latest stable release of FEDOT is on the `master branch <https://github.com/nccr-itmo/FEDOT/tree/master>`__. Make sure you are looking at and working on the actual code if you're looking to contribute code.

The repository includes the following directories:

* Package `core <https://github.com/nccr-itmo/FEDOT/tree/master/core>`__  contains the main classes and scripts. It is a *core* of FEDOT framework
* Package `examples <https://github.com/nccr-itmo/FEDOT/tree/master/examples>`__ includes several *how-to-use-cases* where you can start to discover how FEDOT works
* All *unit tests* can be observed in the `test <https://github.com/nccr-itmo/FEDOT/tree/master/test>`__ directory
* The sources of documentation are in the `docs <https://github.com/nccr-itmo/FEDOT/tree/master/docs>`__

Also you can check `benchmarking <https://github.com/ITMO-NSS-team/AutoML-benchmark>`__ repository that was developed to
show the comparison of FEDOT against the well-known AutoML frameworks.

Basic Concepts
===============

The main process of FEDOT work is *composing* leading to the production of the composite models.

**Composer** is a block that takes meta-requirements and the evolutionary algorithm as an optimization one
and get different chains of models to find the most appropriate solution for the case.

The result of composing and basic object user works with is the Chain:
**Chain** is the tree-based structure of any composite model. It keeps the information of nodes relations
and everything referred to chain properties and restructure.

In fact, any chain has two kinds of nodes:
 - **Primary nodes** are edge (leaf) nodes of the tree where initial case data is located.
 - **Secondary nodes** are all other nodes which transform data during the composing and fitting, including root node with result data.

Meanwhile, every node holds the *Model* which could be ML or any other kind of model.

Current R&D and future plans
============================

At the moment, we execute an extensive set of experiments to determine the most suitable approaches for evolutionary chain optimization, hyperparameters tuning, benchmarking, etc.
The different case studies from different subject areas (metocean science, oil production, seismic, robotics, economics, etc) are in progress now.

The various features are planned to be implemented: multi-data chains, Bayesian networks optimization, domain-specific, equation-based models involvement, interpretable surrogate models, etc.

Any support and contribution are welcome. Our R&D team is open for cooperation with other scientific teams as well as with industrial partners.

Documentation
=============

The common description is available in `FEDOT.Docs <https://itmo-nss-team.github.io/FEDOT.Docs>`__ repository.

Also, detailed FEDOT API description is available in the in `Read the Docs <https://fedot.readthedocs.io/en/latest/>`__.

Contribution Guide
==================

- The contribution guide is available in the `repository <https://github.com/nccr-itmo/FEDOT/blob/master/docs/contributing.rst>`__.

Acknowledgements
================

We acknowledge the contributors for their important impact and the participants of the numerous scientific conferences and workshops for their valuable advice and suggestions.

Contacts
============
- `Telegram channel for solving problems and answering questions on FEDOT <https://t.me/FEDOT_helpdesk>`_
- `Natural System Simulation Team <https://itmo-nss-team.github.io/>`_
- `Anna Kalyuzhnaya <https://scholar.google.com/citations?user=bjiILqcAAAAJ&hl=ru>`_, team leader (anna.kalyuzhnaya@itmo.ru)
- `News feed <https://t.me/NSS_group>`_
- `Youtube channel <https://www.youtube.com/channel/UC4K9QWaEUpT_p3R4FeDp5jA>`_

Supported by
============

- `National Center for Cognitive Research of ITMO University <https://actcognitive.org/>`_

Citation
========

@article{nikitin2020structural,
  title={Structural Evolutionary Learning for Composite Classification Models},
  author={Nikitin, Nikolay O and Polonskaia, Iana S and Vychuzhanin, Pavel and Barabanova, Irina V and Kalyuzhnaya, Anna V},
  journal={Procedia Computer Science},
  volume={178},
  pages={414--423},
  year={2020},
  publisher={Elsevier}}

@inproceedings{kalyuzhnaya2020automatic,
  title={Automatic evolutionary learning of composite models with knowledge enrichment},
  author={Kalyuzhnaya, Anna V and Nikitin, Nikolay O and Vychuzhanin, Pavel and Hvatov, Alexander and Boukhanovsky, Alexander},
  booktitle={Proceedings of the 2020 Genetic and Evolutionary Computation Conference Companion},
  pages={43--44},
  year={2020}}

.. |docs| image:: https://readthedocs.org/projects/ebonite/badge/?style=flat
    :target: https://fedot.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |build| image:: https://github.com/nccr-itmo/FEDOT/workflows/Build/badge.svg?branch=master
    :alt: Build Status
    :target: https://github.com/nccr-itmo/FEDOT/actions

.. |coverage| image:: https://codecov.io/gh/nccr-itmo/FEDOT/branch/master/graph/badge.svg
    :alt: Coverage Status
    :target: https://codecov.io/gh/nccr-itmo/FEDOT

.. |pypi| image:: https://badge.fury.io/py/fedot.svg
    :alt: Supported Python Versions
    :target: https://badge.fury.io/py/fedot

.. |py_6| image:: https://img.shields.io/badge/python_3.6-passing-success
    :alt: Supported Python Versions
    :target: https://img.shields.io/badge/python_3.6-passing-success

.. |py_7| image:: https://img.shields.io/badge/python_3.7-passing-success
    :alt: Supported Python Versions
    :target: https://img.shields.io/badge/python_3.7-passing-success

.. |py_8| image:: https://img.shields.io/badge/python_3.8-passing-success
    :alt: Supported Python Versions
    :target: https://img.shields.io/badge/python_3.8-passing-success

.. |license| image:: https://img.shields.io/github/license/nccr-itmo/FEDOT
    :alt: Supported Python Versions
    :target: https://github.com/nccr-itmo/FEDOT/blob/master/LICENSE.md
