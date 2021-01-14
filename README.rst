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

.. end-badges

This repository contains Fedot - a framework for automated modeling and machine learning. It can build composite models for the different real-world processes in an automated way using an evolutionary approach.

Composite models - the models with heterogeneous graph-based structure, that can consist of ML models, domain-specific models, equation-based models, statistical, and even other composite models. Composite modelling allows obtaining efficient multi-scale solutions for various applied problems.

Fedot can be used for classification, regression, clustering, time series forecasting, and other similar tasks. Also, the derived solutions for other problems (e.g. bayesian generation of synthetic data) can be build using Fedot.Core.

The intro video about Fedot is available here:


.. image:: https://res.cloudinary.com/marcomontalbano/image/upload/v1606396758/video_to_markdown/images/youtube--RjbuV6i6de4-c05b58ac6eb4c4700831b2b3070cd403.jpg
    :target: http://www.youtube.com/watch?v=RjbuV6i6de4
    :alt: Introducing Fedot

   

The project is maintained by the research team of Natural Systems Simulation Lab, which is a part of the National Center for Cognitive Research of ITMO University.

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


FEDOT features
==============
- The generation of high-quality variable-shaped machine learning pipelines for various tasks: binary/multiclass classification, regression, clustering, time series forecasting;
- The structural learning of composite models with different nature (hybrid, bayesian, deep learning, etc) using custom metrics;
- The seamless integration of the custom models (including domain-specific), frameworks and algorithms into pipelines;
- Benchmarking utilities that can run real-world cases (the ready-to-use examples are provided for credit scoring, sea surface height forecasting, oil production forecasting, etc), state-of-the-art-datasets (like PMLB) and synthetic data.


How to use
==========

The main purpose of FEDOT is to identify a suitable composite model for a given dataset.
The model is obtained via optimization process (we also call it 'composing').\
Firstly, you need to prepare datasets for fit and validate and specify a task
that you going to solve:

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

FEDOT API
==========

FEDOT provides a high-level API that allows you to use its capabilities simpler.
At the moment, API can be used for classification and regression tasks only.
But the time series forecasting and clustering support will be implemented soon (you still can solve these tasks via advanced initialization, see above).
Input data must be ether in numpy-array format or CSV files.

To use API, follow these steps:

1. Import Fedot class

.. code-block:: python

  from fedot.api.api_runner import Fedot

2. Select the type of ML-problem and the hyperparameters of Composer (optional).

.. code-block:: python

    task = 'classification'
    composer_params = {'max_depth': 2,
                       'max_arity': 2,
                   'learning_time': 1}

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

The referenced papers:

- Kalyuzhnaya A. V. et al. Automatic evolutionary learning of composite models with knowledge enrichment //Proceedings of the 2020 Genetic and Evolutionary Computation Conference Companion. – 2020. – P. 43-44.
- Kovalchuk S. V. et al. A conceptual approach to complex model management with generalized modelling patterns and evolutionary identification //Complexity. – 2018. – V. 2018.
- Nikitin N. O. et al. Deadline-driven approach for multi-fidelity surrogate-assisted environmental model calibration: SWAN wind wave model case study //Proceedings of the Genetic and Evolutionary Computation Conference Companion. – 2019. – С. 1583-1591.
- Vychuzhanin P., Nikitin N. O., Kalyuzhnaya A. V. Robust Ensemble-Based Evolutionary Calibration of the Numerical Wind Wave Model //International Conference on Computational Science. – Springer, Cham, 2019. – P. 614-627.
- Nikitin N. O. et al. Evolutionary ensemble approach for behavioral credit scoring //International Conference on Computational Science. – Springer, Cham, 2018. – P. 825-831.

Current R&D and future plans
============================

At the moment, we execute an extensive set of experiments to determine the most suitable approaches for evolutionary chain optimization, hyperparameters tuning, benchmarking, etc.
The different case studies from different subject areas (metocean science, oil production, seismic, robotics, economics, etc) are in progress now.
The various features are planned to be implemented: multi-data chains, Bayesian networks optimization, domain-specific, equation-based models involvement, model export and atomization, interpretable surrogate models, etc.

Any support and contribution are welcome.

Documentation
=============

The documentation is available in `FEDOT.Docs <https://itmo-nss-team.github.io/FEDOT.Docs>`__ repository.

The description and source code of underlying algorithms is available in `FEDOT.Algs <https://github.com/ITMO-NSS-team/FEDOT.Algs>`__ repository and its `wiki pages <https://github.com/ITMO-NSS-team/FEDOT.Algs/wiki>`__ (in Russian).

Also, FEDOT API in `Read the Docs <https://fedot.readthedocs.io/en/latest/>`__.

Contribution Guide
==================

- The contribution guide is available in the `repository <https://github.com/nccr-itmo/FEDOT/blob/master/docs/contributing.rst>`__.

Acknowledgements
================

We acknowledge the contributors for their important impact and the participants of the numerous scientific conferences and workshops for their valuable advice and suggestions.

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
