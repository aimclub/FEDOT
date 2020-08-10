FEDOT
============

.. start-badges
.. list-table::
    :stub-columns: 1

    * - package
      - | |py_6| |py_7|
    * - tests
      - | |build| |coveralls|
    * - docs
      - |docs|
    * - license
      - | |license|

.. end-badges

This repository contains the framework for the knowledge-enriched AutoML named FEDOT (Russian: Федот).
It can be used to generate high-quality composite models for the classification, regression, clustering, time series forecasting, and other real-world problems in an automated way.

The project is maintained by the research team of Natural Systems Simulation Lab, which is a part of the National Center for Cognitive Research of ITMO University.

Installation
============
  | git clone https://github.com/nccr-itmo/FEDOT.git
  | cd FEDOT
  | pip install -r requirements.txt

FEDOT features
==============
.. Здесь перечислим буллетпоинты, что (а) Федот позволяет делать и (б) что он делает хорошо

To combine the proposed concepts and methods with the existing state-of-the-art approaches and share the obtained experience with the community, we decided to develop the FEDOT framework.

The framework kernel can be configured for different classes of tasks. The framework includes the library with implementations of intelligent algorithms to identify data-driven models with different requirements; and composite models (chains of models) for solving specific subject tasks (social and financial, metocean, physical, etc.).

It is possible to obtain models with given parameters of quality, complexity, interpretability; to get an any-time result; to pause and resume model identification; to integrate many popular Python open source solutions for AutoML/meta-learning, optimization, quality assessment, etc.; re-use the models created by other users.


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
  metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC

Next, you need to specify requirements for composer.
In this case, GPComposer is chosen that is based on evolutionary algorithm.

.. code-block:: python

  composer_requirements = GPComposerRequirements(
    primary=available_model_types,
    secondary=available_model_types, max_arity=3,
    max_depth=3, pop_size=20, num_of_generations=20,
    crossover_prob=0.8, mutation_prob=0.8, max_lead_time=20)
  composer = GPComposer()

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
  print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')

Extended examples:
- Credit scoring problem, i.e. binary classification task - `here <https://github.com/nccr-itmo/FEDOT/blob/master/cases/credit_scoring_problem.py>`__
- Time series forecasting, i.e. regression - `here <https://github.com/nccr-itmo/FEDOT/blob/master/cases/metocean_forecasting_problem.py>`__

.. Можно добавить ссылок на видео туториалы, еще что-то

Project structure
=================

.. Здесь, наверное, можно коротко описать основные модули, архитектуру и сослаться на документацию


Basic Concepts
==============

.. Здесь можно описать основные термины и концепты, которые мы включаем в FEDOT: композер, primary/secondary узлы, цепочки,

.. можно добавить ссылки на всякие наши статьи


Current R&D and future plans
============================

.. Здесь можно коротко описать, над чем мы сейчас работаем

Documentation
=============

The documentation is available in `FEDOT.Docs <https://itmo-nss-team.github.io/FEDOT.Docs>`__ repository.

The description and source code of underlying algorithms is available in `FEDOT.Algs <https://github.com/ITMO-NSS-team/FEDOT.Algs>`__ repository and its `wiki pages <https://github.com/ITMO-NSS-team/FEDOT.Algs/wiki>`__ (in Russian).

.. Дополнительно будет ссылка на readthedocs

Contribution Guide
==================

Acknowledgements
================

Supported by
============

- `National Center for Cognitive Research of ITMO University <https://actcognitive.org/>`_

Citation
========

latex-ссылка на основную статью про фреймворк (когда она появится)

.. |docs| image:: https://readthedocs.org/projects/ebonite/badge/?style=flat
    :target: https://itmo-nss-team.github.io/FEDOT.Docs/
    :alt: Documentation Status

.. |build| image:: https://travis-ci.com/nccr-itmo/FEDOT.svg?token=ABTJ8bEXZokRxF3wLrtJ&branch=master
    :alt: Build Status
    :target: https://travis-ci.com/nccr-itmo/FEDOT

.. |coveralls| image:: https://coveralls.io/repos/github/nccr-itmo/FEDOT/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://coveralls.io/github/nccr-itmo/FEDOT?branch=master

.. |py_6| image:: https://img.shields.io/badge/python_3.6-passing-success
    :alt: Supported Python Versions
    :target: https://img.shields.io/badge/python_3.6-passing-success

.. |py_7| image:: https://img.shields.io/badge/python_3.7-passing-success
    :alt: Supported Python Versions
    :target: https://img.shields.io/badge/python_3.7-passing-success

.. |license| image:: https://img.shields.io/github/license/nccr-itmo/FEDOT
    :alt: Supported Python Versions
    :target: https://img.shields.io/github/license/nccr-itmo/FEDOT