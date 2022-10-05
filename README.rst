.. image:: docs/fedot_logo.png
   :alt: Logo of FEDOT framework

.. start-badges
.. list-table::
   :stub-columns: 1

   * - package
     - | |pypi| |py_7| |py_8| |py_9|
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

**FEDOT** is an open-source framework for automated modeling and machine learning (AutoML) problems. This framework is distributed under the 3-Clause BSD license.

It provides automatic generative design of machine learning pipelines for various real-world problems. The core of FEDOT is based on an evolutionary approach and supports classification (binary and multiclass), regression, clustering, and time series prediction problems.

.. image:: docs/pipeline_small.png
   :alt: The structure of the modeling pipeline that can be optimised by FEDOT

The key feature of the framework is the complex management of interactions between the various blocks of pipelines. It is represented as a graph that defines connections between data preprocessing and model blocks.

The project is maintained by the research team of the Natural Systems Simulation Lab, which is a part of the `National Center for Cognitive Research of ITMO University <https://actcognitive.org/>`__.

More details about FEDOT are available in the next video:


.. image:: https://res.cloudinary.com/marcomontalbano/image/upload/v1606396758/video_to_markdown/images/youtube--RjbuV6i6de4-c05b58ac6eb4c4700831b2b3070cd403.jpg
   :target: http://www.youtube.com/watch?v=RjbuV6i6de4
   :alt: Introducing Fedot

FEDOT Features
==============

The main features of the framework are follows:

- **Flexibility.** FEDOT is highly flexible: it can be used to automate the construction of solutions for various problems, data types, and models;
- **Integration with ML libraries.** FEDOT supports widely used ML libraries (Scikit-Learn, Catboost, Xgboost, etc.) and allows you to integrate custom ones;
- **Extensibility for new domains.** Pipeline optimization algorithms are data- and task-independent, yet you can use special templates for a specific task class or data type (time series forecasting, NLP, tabular data, etc.) to increase the efficiency;
- **No limits.** The framework is versatile and not limited to specific modeling tasks, for example, it can be use in ODE or PDE;
- **Support of hyper-parameter tuning.** Hyper-parameters tuning methods are supported. Custom methods can also be integrated in FEDOT;
- **Reproducibility.** You can export the resulting pipelines in JSON format for experiment reproducibility.

Compared to other frameworks:

- There are no limits to specific modeling tasks, therefore FEDOT claims versatility and expandability;
- Allows managing the complexity of models and thereby achieving better results.
- Allows building pipelines using different types of input data (texts, images, tables, etc.) and consisting of various models.

Installation
============

The simplest way to install FEDOT is using ``pip``:

.. code-block::

  $ pip install fedot

Installation with optional dependencies for image and text processing, and for DNNs:

.. code-block::

  $ pip install fedot[extra]

How to Use
==========

FEDOT provides a high-level API that allows you to use its capabilities in a simple way. The API can be used for classification, regression, and time series forecasting problems.

To use the API, follow these steps:

1. Import ``Fedot`` class

.. code-block:: python

 from fedot.api.main import Fedot

2. Initialize the Fedot object and define the type of modeling problem. It provides a fit/predict interface:

- ``Fedot.fit()`` begins the optimization and returns the resulting composite pipeline;
- ``Fedot.predict()`` predicts target values for the given input data using already fitted pipeline;
- ``Fedot.get_metrics()`` estimates the quality of predictions using selected metrics.

NumPy arrays, Pandas DataFrames, and the file's path can be used as sources of input data. In case below, `x_train`, `y_train` and `x_test` are `numpy.ndarray()`:

.. code-block:: python

    model = Fedot(problem='classification', timeout=5, preset='best_quality', n_jobs=-1)
    model.fit(features=x_train, target=y_train)
    prediction = model.predict(features=x_test)
    metrics = model.get_metrics(target=y_test)

More information about the API is available in `documentation <https://fedot.readthedocs.io/en/latest/api/api.html>`__ and advanced approaches are in `Examples & Tutorials <https://github.com/nccr-itmo/FEDOT#examples--tutorials>`__ section.

Examples & Tutorials
====================

Jupyter notebooks with tutorials are located in the `examples repository <https://github.com/ITMO-NSS-team/fedot-examples>`__. There you can find the following guides:

* `Intro to AutoML <https://github.com/ITMO-NSS-team/fedot-examples/blob/main/notebooks/latest/1_intro_to_automl.ipynb>`__
* `Intro to FEDOT functionality <https://github.com/ITMO-NSS-team/fedot-examples/blob/main/notebooks/latest/2_intro_to_fedot.ipynb>`__
* `Intro to time series forecasting with FEDOT <https://github.com/ITMO-NSS-team/fedot-examples/blob/main/notebooks/latest/3_intro_ts_forecasting.ipynb>`__
* `Advanced time series forecasting <https://github.com/ITMO-NSS-team/fedot-examples/blob/main/notebooks/latest/4_auto_ts_forecasting.ipynb>`__
* `Gap-filling in time series and out-of-sample forecasting <https://github.com/ITMO-NSS-team/fedot-examples/blob/main/notebooks/latest/5_ts_specific_cases.ipynb>`__
* `Hybrid modelling with custom models <https://github.com/ITMO-NSS-team/fedot-examples/blob/main/notebooks/latest/6_hybrid_modelling.ipynb>`__

Notebooks are issued with the corresponding release versions (the default version is 'latest').

Also, external examples are available:

* `Kaggle: baseline for Microsoft Stock - Time Series Analysis task <https://www.kaggle.com/dreamlone/microsoft-stocks-price-prediction-automl>`__

Extended examples:

- Credit scoring problem, i.e. `binary classification task <https://github.com/nccr-itmo/FEDOT/blob/master/cases/credit_scoring/credit_scoring_problem.py>`__
- Time series forecasting, i.e. `random process regression <https://github.com/nccr-itmo/FEDOT/blob/master/cases/metocean_forecasting_problem.py>`__
- Spam detection, i.e. `natural language preprocessing <https://github.com/nccr-itmo/FEDOT/blob/master/cases/spam_detection.py>`__
- Movie rating prediction with `multi-modal data <https://github.com/nccr-itmo/FEDOT/blob/master/cases/multi_modal_rating_prediction.py>`__


Also, several video tutorials are `available <https://www.youtube.com/playlist?list=PLlbcHj5ytaFUjAxpZf7FbEaanmqpDYhnc>`__ (in Russian).

Publications About FEDOT
========================

We also published several posts and news devoted to the different aspects of the framework:

In English:

- How AutoML helps to create composite AI? - `towardsdatascience.com <https://towardsdatascience.com/how-automl-helps-to-create-composite-ai-f09e05287563>`__
- AutoML for time series: definitely a good idea - `towardsdatascience.com <https://towardsdatascience.com/automl-for-time-series-definitely-a-good-idea-c51d39b2b3f>`__
- AutoML for time series: advanced approaches with FEDOT framework - `towardsdatascience.com <https://towardsdatascience.com/automl-for-time-series-advanced-approaches-with-fedot-framework-4f9d8ea3382c>`__
- Winning a flood-forecasting hackathon with hydrology and AutoML - `towardsdatascience.com <https://towardsdatascience.com/winning-a-flood-forecasting-hackathon-with-hydrology-and-automl-156a8a7a4ede>`__
- Clean AutoML for “Dirty” Data - `towardsdatascience.com <https://towardsdatascience.com/clean-automl-for-dirty-data-how-and-why-to-automate-preprocessing-of-tables-in-machine-learning-d79ac87780d3>`__
- FEDOT as a factory of human-competitive results - `youtube.com <https://www.youtube.com/watch?v=9Rhqcsrolb8&ab_channel=NSS-Lab>`__
- Hyperparameters Tuning for Machine Learning Model Ensembles - `towardsdatascience.com <https://towardsdatascience.com/hyperparameters-tuning-for-machine-learning-model-ensembles-8051782b538b>`__

In Russian:

- Как AutoML помогает создавать модели композитного ИИ — говорим о структурном обучении и фреймворке FEDOT - `habr.com <https://habr.com/ru/company/spbifmo/blog/558450>`__
- Прогнозирование временных рядов с помощью AutoML - `habr.com <https://habr.com/ru/post/559796/>`__
- Как мы “повернули реки вспять” на Emergency DataHack 2021, объединив гидрологию и AutoML - `habr.com <https://habr.com/ru/post/577886/>`__
- Чистый AutoML для “грязных” данных: как и зачем автоматизировать предобработку таблиц в машинном обучении - `ODS blog <https://habr.com/ru/company/ods/blog/657525/>`__
- Фреймворк автоматического машинного обучения FEDOT (Конференция Highload++ 2022) - `presentation <https://docs.yandex.ru/docs/view?url=ya-disk-public%3A%2F%2Fi27LScu3s3IIHDzIXt9O5EiEAMl6ThY6QLu3X1oYH%2FFiAl%2BLcNp4O4yTSYd2gRZnW5aDQ4kMZEXE%2BwNjbq78ug%3D%3D%3A%2F%D0%94%D0%B5%D0%BD%D1%8C%201%2F4.%D0%A1%D0%B8%D0%BD%D0%BD%D0%B0%D0%BA%D1%81%2F9.Open%20source-%D1%82%D1%80%D0%B8%D0%B1%D1%83%D0%BD%D0%B0_HL_FEDOT.pptx&name=9.Open%20source-%D1%82%D1%80%D0%B8%D0%B1%D1%83%D0%BD%D0%B0_HL_FEDOT.pptx>`__
- Про настройку гиперпараметров ансамблей моделей машинного обучения - `habr.com <https://habr.com/ru/post/672486/>`__

In Chinese:

- 生成式自动机器学习系统 (presentation at the "Open Innovations 2.0" conference) - `youtube.com <https://www.youtube.com/watch?v=PEET0EbCSCY>`__


Project Structure
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

Currently, we are working on new features and trying to improve the performance and the user experience of FEDOT.
The major ongoing tasks and plans:

* Effective and ready-to-use pipeline templates for certain tasks and data types;
* Integration with GPU via Rapids framework;
* Alternative optimization methods of fixed-shaped pipelines;
* Integration with MLFlow for import and export of the pipelines;
* Improvement of high-level API.


Also, we are doing several research tasks related to AutoML time-series benchmarking and multi-modal modeling.

Any contribution is welcome. Our R&D team is open for cooperation with other scientific teams as well as with industrial partners.

Documentation
=============

The general description is available in `FEDOT.Docs <https://itmo-nss-team.github.io/FEDOT.Miscellaneous>`__ repository.

Also, a detailed FEDOT API description is available in the `Read the Docs <https://fedot.readthedocs.io/en/latest/>`__.

Contribution Guide
==================

- The contribution guide is available in the `repository <https://github.com/nccr-itmo/FEDOT/blob/master/docs/source/contribution.rst>`__.

Acknowledgments
===============

We acknowledge the contributors for their important impact and the participants of the numerous scientific conferences and workshops for their valuable advice and suggestions.

Side Projects
=============
- The prototype of web-GUI for FEDOT is available in `FEDOT.WEB <https://github.com/nccr-itmo/FEDOT.Web>`__ repository.


Contacts
========
- `Telegram channel for solving problems and answering questions on FEDOT <https://t.me/FEDOT_helpdesk>`_
- `Natural System Simulation Team <https://itmo-nss-team.github.io/>`_
- `Anna Kalyuzhnaya <https://scholar.google.com/citations?user=bjiILqcAAAAJ&hl=ru>`_, Team leader (anna.kalyuzhnaya@itmo.ru)
- `Newsfeed <https://t.me/NSS_group>`_
- `Youtube channel <https://www.youtube.com/channel/UC4K9QWaEUpT_p3R4FeDp5jA>`_

Supported by
============

- `National Center for Cognitive Research of ITMO University <https://actcognitive.org/>`_

Citation
========

@article{nikitin2021automated,
  title = {Automated evolutionary approach for the design of composite machine learning pipelines},
  author = {Nikolay O. Nikitin and Pavel Vychuzhanin and Mikhail Sarafanov and Iana S. Polonskaia and Ilia Revin and Irina V. Barabanova and Gleb Maximov and Anna V. Kalyuzhnaya and Alexander Boukhanovsky},
  journal = {Future Generation Computer Systems},
  year = {2021},
  issn = {0167-739X},
  doi = {https://doi.org/10.1016/j.future.2021.08.022}}

@inproceedings{polonskaia2021multi,
  title={Multi-Objective Evolutionary Design of Composite Data-Driven Models},
  author={Polonskaia, Iana S. and Nikitin, Nikolay O. and Revin, Ilia and Vychuzhanin, Pavel and Kalyuzhnaya, Anna V.},
  booktitle={2021 IEEE Congress on Evolutionary Computation (CEC)},
  year={2021},
  pages={926-933},
  doi={10.1109/CEC45853.2021.9504773}}


Other papers - in `ResearchGate <https://www.researchgate.net/project/Evolutionary-multi-modal-AutoML-with-FEDOT-framework>`_.

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

.. |py_7| image:: https://img.shields.io/badge/python_3.7-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.7-passing-success

.. |py_8| image:: https://img.shields.io/badge/python_3.8-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.8-passing-success

.. |py_9| image:: https://img.shields.io/badge/python_3.9-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.9-passing-success

.. |license| image:: https://img.shields.io/github/license/nccr-itmo/FEDOT
   :alt: Supported Python Versions
   :target: https://github.com/nccr-itmo/FEDOT/blob/master/LICENSE.md

.. |downloads_stats| image:: https://static.pepy.tech/personalized-badge/fedot?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
   :target: https://pepy.tech/project/fedot

.. |tg| image:: https://img.shields.io/badge/Telegram-Group-blue.svg
          :target: https://t.me/FEDOT_helpdesk
          :alt: Telegram Chat
