.. |eng| image:: https://img.shields.io/badge/lang-en-red.svg
   :target: /README_en.rst

.. |rus| image:: https://img.shields.io/badge/lang-ru-yellow.svg
   :target: /README.rst

.. image:: /docs/fedot_logo.png
   :alt: Logo of FEDOT framework

.. start-badges
.. list-table::
   :stub-columns: 1

   * - package
     - | |pypi| |python|
   * - tests
     - | |build| |integration| |coverage|
   * - docs
     - |docs|
   * - license
     - | |license|
   * - stats
     - | |downloads_stats|
   * - support
     - | |tg|
   * - languages
     - | |eng| |rus|
   * - mirror
     - | |gitlab|
   * - funding
     - | |ITMO| |NCCR|
.. end-badges

**FEDOT** is an open-source framework for automated modeling and machine learning (AutoML) problems. This framework is distributed under the 3-Clause BSD license.

It provides automatic generative design of machine learning pipelines for various real-world problems. The core of FEDOT is based on an evolutionary approach and supports classification (binary and multiclass), regression, clustering, and time series prediction problems.

.. image:: /docs/fedot-workflow.png
   :alt: The structure of the AutoML workflow in FEDOT

The key feature of the framework is the complex management of interactions between various blocks of pipelines. It is represented as a graph that defines connections between data preprocessing and model blocks.

The project is maintained by the research team of the Natural Systems Simulation Lab, which is a part of the `National Center for Cognitive Research of ITMO University <https://actcognitive.org/>`__.

More details about FEDOT are available in the next video:


.. image:: https://res.cloudinary.com/marcomontalbano/image/upload/v1606396758/video_to_markdown/images/youtube--RjbuV6i6de4-c05b58ac6eb4c4700831b2b3070cd403.jpg
   :target: http://www.youtube.com/watch?v=RjbuV6i6de4
   :alt: Introducing Fedot

FEDOT concepts
==============

- **Flexibility.** FEDOT can be used to automate the construction of solutions for various `problems <https://fedot.readthedocs.io/en/master/introduction/fedot_features/main_features.html#involved-tasks>`_, `data types <https://fedot.readthedocs.io/en/master/introduction/fedot_features/automation_features.html#data-nature>`_ (texts, images, tables), and `models <https://fedot.readthedocs.io/en/master/advanced/automated_pipelines_design.html>`_;
- **Extensibility.** Pipeline optimization algorithms are data- and task-independent, yet you can use `special strategies <https://fedot.readthedocs.io/en/master/api/strategies.html>`_ for specific tasks or data types (time-series forecasting, NLP, tabular data, etc.) to increase the efficiency;
- **Integrability.** FEDOT supports widely used ML libraries (Scikit-learn, CatBoost, XGBoost, etc.) and allows you to integrate `custom ones <https://fedot.readthedocs.io/en/master/api/strategies.html#module-fedot.core.operations.evaluation.custom>`_;
- **Tuningability.** Various `hyper-parameters tuning methods <https://fedot.readthedocs.io/en/master/advanced/hyperparameters_tuning.html>`_ are supported including models' custom evaluation metrics and search spaces;
- **Versatility.** FEDOT is `not limited to specific modeling tasks <https://fedot.readthedocs.io/en/master/advanced/architecture.html>`_, for example, it can be used in ODE or PDE;
- **Reproducibility.** Resulting pipelines can be `exported separately as JSON <https://fedot.readthedocs.io/en/master/advanced/pipeline_import_export.html>`_ or `together with your input data as ZIP archive <https://fedot.readthedocs.io/en/master/advanced/project_import_export.html>`_ for experiments reproducibility;
- **Customizability.** FEDOT allows `managing models complexity <https://fedot.readthedocs.io/en/master/introduction/fedot_features/automation_features.html#models-used>`_ and thereby achieving desired quality.

Installation
============

- Package installer for Python **pip**

The simplest way to install FEDOT is using ``pip``:

.. code-block::

  $ pip install fedot

Installation with optional dependencies for image and text processing, and for DNNs:

.. code-block::

  $ pip install fedot[extra]

- **Docker** container

Available docker images can be found here `here <https://github.com/aimclub/FEDOT/tree/master/docker/README_en.rst>`_.

How to Use
==========

FEDOT provides a high-level API that allows you to use its capabilities in a simple way. The API can be used for classification, regression, and time series forecasting problems.

To use the API, follow these steps:

1. Import ``Fedot`` class

.. code-block:: python

 from fedot.api.main import Fedot

2. Initialize the Fedot object and define the type of modeling problem. It provides a fit/predict interface:

- ``Fedot.fit()`` begins the optimization and returns the resulting composite pipeline;
- ``Fedot.predict()`` predicts target values for the given input data using an already fitted pipeline;
- ``Fedot.get_metrics()`` estimates the quality of predictions using selected metrics.

NumPy arrays, Pandas DataFrames, and the file's path can be used as sources of input data. In the case below, ``x_train``, ``y_train`` and ``x_test`` are ``numpy.ndarray()``:

.. code-block:: python

    model = Fedot(problem='classification', timeout=5, preset='best_quality', n_jobs=-1)
    model.fit(features=x_train, target=y_train)
    prediction = model.predict(features=x_test)
    metrics = model.get_metrics(target=y_test)

More information about the API is available in the `documentation <https://fedot.readthedocs.io/en/latest/api/api.html>`__ section and advanced approaches are in the `advanced <https://github.com/aimclub/FEDOT/tree/master/examples/advanced>`__ section.

Examples & Tutorials
====================

Jupyter notebooks with tutorials are located in `examples repository <https://github.com/ITMO-NSS-team/fedot-examples>`__. There you can find the following guides:

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

- Credit scoring problem, i.e. `binary classification task <https://github.com/aimclub/FEDOT/blob/master/cases/credit_scoring/credit_scoring_problem.py>`__
- Time series forecasting, i.e. `random process regression <https://github.com/aimclub/FEDOT/blob/master/cases/metocean_forecasting_problem.py>`__
- Spam detection, i.e. `natural language preprocessing <https://github.com/aimclub/FEDOT/blob/master/cases/spam_detection.py>`__
- Wine variety prediction with `multi-modal data <https://github.com/aimclub/FEDOT/blob/master/examples/advanced/multimodal_text_num_example.py>`__


Also, several video tutorials are available `available <https://www.youtube.com/playlist?list=PLlbcHj5ytaFUjAxpZf7FbEaanmqpDYhnc>`__ (in Russian).

Publications About FEDOT
========================

We also published several posts devoted to different aspects of the framework:

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

The latest stable release of FEDOT is in the `master branch <https://github.com/aimclub/FEDOT/tree/master>`__.

The repository includes the following directories:

* Package `core <https://github.com/aimclub/FEDOT/tree/master/fedot/core>`__  contains the main classes and scripts. It is the *core* of the FEDOT framework
* Package `examples <https://github.com/aimclub/FEDOT/tree/master/examples>`__ includes several *how-to-use-cases* where you can start to discover how FEDOT works
* All *unit and integration tests* can be observed in the `test <https://github.com/aimclub/FEDOT/tree/master/test>`__ directory
* The sources of the documentation are in the `docs <https://github.com/aimclub/FEDOT/tree/master/docs>`__ directory

Current R&D and future plans
============================

Currently, we are working on new features and trying to improve the performance and the user experience of FEDOT.
The major ongoing tasks and plans:

* Implementation of large language model for AutoML tasks in `FEDOT.LLM <https://github.com/ITMO-NSS-team/FEDOT.LLM>`__ repository.
* Implementation of meta-learning based at GNN and RL (see `GAMLET <https://github.com/ITMO-NSS-team/GAMLET>`__)
* Improvement of the optimisation-related algorithms implemented in `GOLEM <https://github.com/aimclub/GOLEM/>`__.
* Support for more complicated pipeline design patters, especially for time series forecasting.

In addition, we are working on a number of research tasks related to benchmarking of time series forecasting using AutoML and multimodal modelling.

Any contribution is welcome. Our R&D team is open for cooperation with other scientific teams as well as with industrial partners.

Documentation
=============

Also, a detailed FEDOT API description is available in `Read the Docs <https://fedot.readthedocs.io/en/latest/>`__.

Contribution Guide
==================

- The contribution guide is available in this `repository <https://github.com/aimclub/FEDOT/blob/master/docs/source/contribution.rst>`__.

Acknowledgments
===============

We acknowledge the contributors for their important impact and the participants of numerous scientific conferences and workshops for their valuable advice and suggestions.

Side Projects
=============
- The optimisation core implemented in the GOLEM.
- The prototype of the web-GUI for FEDOT is available in the `FEDOT.WEB <https://github.com/aimclub/FEDOT.Web>`__ repository.
- The prototype of FEDOT-based meta-AutoML in the GAMLET.
- The prototype of FEDOT-based LLM for nexgen AutoML in the FEDOT.LLM.

Contacts
========
- `Telegram channel for solving problems and answering questions about FEDOT <https://t.me/FEDOT_helpdesk>`_
- `Natural System Simulation Team <https://itmo-nss-team.github.io/>`_
- `Anna Kalyuzhnaya <https://scholar.google.com/citations?user=bjiILqcAAAAJ&hl=ru>`_, Team leader (anna.kalyuzhnaya@itmo.ru)
- `Telegram newsfeed channel <https://t.me/NSS_group>`_
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

.. |docs| image:: https://readthedocs.org/projects/ebonite/badge/?style=flat
   :target: https://fedot.readthedocs.io/en/latest/
   :alt: Documentation Status

.. |build| image:: https://github.com/aimclub/FEDOT/actions/workflows/unit-build.yml/badge.svg
   :alt: Build Status
   :target: https://github.com/aimclub/FEDOT/actions/workflows/unit-build.yml

.. |integration| image:: https://github.com/aimclub/FEDOT/actions/workflows/integration-build.yml/badge.svg
   :alt: Integration Build Status
   :target: https://github.com/aimclub/FEDOT/actions/workflows/integration-build.yml

.. |coverage| image:: https://codecov.io/gh/aimclub/FEDOT/branch/master/graph/badge.svg
   :alt: Coverage Status
   :target: https://codecov.io/gh/aimclub/FEDOT

.. |pypi| image:: https://badge.fury.io/py/fedot.svg
   :alt: Supported Python Versions
   :target: https://badge.fury.io/py/fedot

.. |python| image:: https://img.shields.io/pypi/pyversions/fedot.svg
   :alt: Supported Python Versions
   :target: https://img.shields.io/pypi/pyversions/fedot

.. |license| image:: https://img.shields.io/github/license/aimclub/FEDOT
   :alt: Supported Python Versions
   :target: https://github.com/aimclub/FEDOT/blob/master/LICENSE.md

.. |downloads_stats| image:: https://static.pepy.tech/personalized-badge/fedot?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
   :target: https://pepy.tech/project/fedot

.. |tg| image:: https://img.shields.io/badge/Telegram-Group-blue.svg
   :target: https://t.me/FEDOT_helpdesk
   :alt: Telegram Chat

.. |ITMO| image:: https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge_rus.svg
   :alt: Acknowledgement to ITMO
   :target: https://itmo.ru

.. |NCCR| image:: https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/NCCR_badge.svg
   :alt: Acknowledgement to NCCR
   :target: https://actcognitive.org/

.. |gitlab| image:: https://img.shields.io/badge/mirror-GitLab-orange
   :alt: GitLab mirror for this repository
   :target: https://gitlab.actcognitive.org/itmo-nss-team/FEDOT
