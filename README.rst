FEDOT
=====

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
   * - stats
     - | |downloads_stats|
   * - support
     - | |tg|


.. end-badges

This repository contains FEDOT - an open-source framework for automated modeling and machine learning (AutoML). It can build custom modeling pipelines for different real-world processes in an automated way using an evolutionary approach. FEDOT supports classification (binary and multiclass), regression, clustering, and time series prediction tasks.

.. image:: https://itmo-nss-team.github.io/FEDOT.Docs/img/pipeline_small.png
   :alt: The structure of the modeling pipeline that can be optimised by FEDOT

The main feature of the framework is the complex management of interactions between various blocks of pipelines. First of all, this includes the stage of machine learning model design. FEDOT allows you to not just choose the best type of the model, but to create a complex (composite) model. It allows you to combine several models of different complexity, which helps you to achieve better modeling quality than when using any of these models separately. Within the framework, we describe composite models in the form of a graph defining the connections between data preprocessing blocks and model blocks.

The framework is not limited to specific AutoML tasks (such as pre-processing of input data, feature selection, or optimization of model hyperparameters), but allows you to solve a more general structural learning problem - for a given data set, a solution is built in the form of a graph (DAG), the nodes of which are represented by ML models, pre-processing procedures, and data transformation.

The project is maintained by the research team of the Natural Systems Simulation Lab, which is a part of the National Center for Cognitive Research of ITMO University.


The intro video about Fedot is available here:


.. image:: https://res.cloudinary.com/marcomontalbano/image/upload/v1606396758/video_to_markdown/images/youtube--RjbuV6i6de4-c05b58ac6eb4c4700831b2b3070cd403.jpg
   :target: http://www.youtube.com/watch?v=RjbuV6i6de4
   :alt: Introducing Fedot

FEDOT features
==============

The main features of the framework are as follows:

- The FEDOT architecture is highly flexible and therefore the framework can be used to automate the creation of mathematical models for various problems, types of data, and models;
- FEDOT already supports popular ML libraries (scikit-learn, keras, statsmodels, etc.), but you can also integrate custom tools into the framework if necessary;
- Pipeline optimization algorithms are not tied to specific data types or tasks, but you can use special templates for a specific task class or data type (time series forecasting, NLP, tabular data, etc.) to increase the efficiency;
- The framework is not limited only to machine learning, it is possible to embed models related to specific areas into pipelines (for example, models in ODE or PDE);
- Additional methods for hyperparameters tuning can be seamlessly integrated into FEDOT (in addition to those already supported);
- The resulting pipelines can be exported in a human-readable JSON format, which allows you to achieve reproducibility of the experiments.

Thus, compared to other frameworks, FEDOT:

- Is not limited to specific modeling tasks and claims versatility and expandability;
- Allows managing the complexity of models and thereby achieving better results.
- Allows building models using input data of various nature (texts, images, tables, etc.) and consisting of different types of models.

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
============================

FEDOT provides a high-level API that allows you to use its capabilities in a simple way.
At the moment, the API can be used for classification and regression tasks only.
But the time series forecasting and clustering support will be implemented soon (you can still solve these tasks via advanced initialization, see below).
Input data must be either in NumPy arrays or CSV files.

To use the API, follow these steps:

1. Import Fedot class

.. code-block:: python

 from fedot.api.main import Fedot

2. Initialize the Fedot object and define the type of modeling problem. It provides a fit/predict interface:

- fedot.fit runs the optimization and returns the resulting composite model;
- fedot.predict returns the prediction for the given input data;
- fedot.get_metrics estimates the quality of predictions using selected metrics

Numpy arrays, pandas data frames, and file paths can be used as sources of input data.

.. code-block:: python

 model = Fedot(problem='classification')

 model.fit(features=train_data.features, target=train_data.target)
 prediction = model.predict(features=test_data.features)

 metrics = auto_model.get_metrics()

How to use (advanced approach)
==============================

The main purpose of FEDOT is to identify a suitable composite model for a given dataset.
The model is obtained via an optimization process (we also call it 'composing') that can be configured in a more detailed way if necessary.
Firstly, you need to prepare datasets for composing and validation and specify a task that you are going to solve:

.. code-block:: python

 task = Task(TaskTypesEnum.classification)
 dataset_to_compose = InputData.from_csv(train_file_path, task=task)
 dataset_to_validate = InputData.from_csv(test_file_path, task=task)

Then, choose a set of models that can be included in the composite model and the optimized metric function:

.. code-block:: python

 available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)
 metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

Next, you need to specify the requirements for the composer.
In this case, a GPComposer that is based on an evolutionary algorithm is chosen.

.. code-block:: python

 composer_requirements = GPComposerRequirements(
   primary=available_model_types,
   secondary=available_model_types, max_arity=3,
   max_depth=3, pop_size=20, num_of_generations=20,
   crossover_prob=0.8, mutation_prob=0.8, max_lead_time=20)

After that you need to initialize the composer with the builder using the specified parameters:

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

Finally, you can test the resulting model on the validation dataset:

.. code-block:: python

 roc_on_valid_evo_composed = calculate_validation_metric(chain_evo_composed,
                                                         dataset_to_validate)
 print(f'Composed ROC AUC is {roc_on_valid_evo_composed}')


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

The latest stable release of FEDOT is on the `master branch <https://github.com/nccr-itmo/FEDOT/tree/master>`__.

The repository includes the following directories:

* Package `core <https://github.com/nccr-itmo/FEDOT/tree/master/fedot/core>`__  contains the main classes and scripts. It is the *core* of FEDOT framework
* Package `examples <https://github.com/nccr-itmo/FEDOT/tree/master/examples>`__ includes several *how-to-use-cases* where you can start to discover how FEDOT works
* All *unit and integration tests* can be observed in the `test <https://github.com/nccr-itmo/FEDOT/tree/master/test>`__ directory
* The sources of the documentation are in the `docs <https://github.com/nccr-itmo/FEDOT/tree/master/docs>`__

Also, you can check `benchmarking <https://github.com/ITMO-NSS-team/FEDOT-benchmarks>`__ a repository that was developed to provide a comparison of FEDOT against some well-known AutoML frameworks.

Current R&D and future plans
============================

At the moment, we are executing an extensive set of experiments to determine the most suitable approaches for evolutionary chain optimization, hyperparameters tuning, benchmarking, etc.
The different case studies from different subject areas (metocean science, geology, robotics, economics, etc) are in progress now.

Various features are planned to be implemented: multi-data chains, Bayesian networks optimization, domain-specific and equation-based models, interpretable surrogate models, etc.

Any contribution is welcome. Our R&D team is open for cooperation with other scientific teams as well as with industrial partners.

Documentation
=============

The general description is available in `FEDOT.Docs <https://itmo-nss-team.github.io/FEDOT.Docs>`__ repository.

Also, a detailed FEDOT API description is available in the `Read the Docs <https://fedot.readthedocs.io/en/latest/>`__.

Contribution Guide
==================

- The contribution guide is available in the `repository <https://github.com/nccr-itmo/FEDOT/blob/master/docs/contributing.rst>`__.

Acknowledgments
================

We acknowledge the contributors for their important impact and the participants of the numerous scientific conferences and workshops for their valuable advice and suggestions.

Contacts
========
- `Telegram channel for solving problems and answering questions on FEDOT <https://t.me/FEDOT_helpdesk>`_
- `Natural System Simulation Team <https://itmo-nss-team.github.io/>`_
- `Anna Kalyuzhnaya <https://scholar.google.com/citations?user=bjiILqcAAAAJ&hl=ru>`_, team leader (anna.kalyuzhnaya@itmo.ru)
- `Newsfeed <https://t.me/NSS_group>`_
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

.. |downloads_stats| image:: https://static.pepy.tech/personalized-badge/fedot?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
   :target: https://pepy.tech/project/fedot

.. |tg| image:: https://img.shields.io/badge/Telegram-Group-blue.svg
          :target: https://t.me/FEDOT_helpdesk
          :alt: Telegram Chat
