# FEDOT
[![Build Status](https://travis-ci.com/J3FALL/THEODOR.svg?token=ABTJ8bEXZokRxF3wLrtJ&branch=master)](https://travis-ci.com/J3FALL/THEODOR) [![Coverage Status](https://coveralls.io/repos/github/J3FALL/THEODOR/badge.svg?branch=master)](https://coveralls.io/github/J3FALL/THEODOR?branch=master)

This repository contains the framework for the knowledge-enriched AutoML named FEDOT (Russian: Федот).
It can be used to generate high-quality composite models for the classification, regression, clustering, time series forecasting, and other real-world problems in an automated way.

The project is maintained by the research team of Natural Systems Simulation Lab, which is a part of the National Center for Cognitive Research of ITMO University.


## Short description

To combine the proposed concepts and methods with the existing state-of-the-art approaches and share the obtained experience with the community, we decided to develop the FEDOT framework.

The framework kernel can be configured for different classes of tasks. The framework includes the library with implementations of intelligent algorithms to identify data-driven models with different requirements; and composite models (chains of models) for solving specific subject tasks (social and financial, metocean, physical, etc.).

It is possible to obtain models with given parameters of quality, complexity, interpretability; to get an any-time result; to pause and resume model identification; to integrate many popular Python open source solutions for AutoML/meta-learning, optimization, quality assessment, etc.; re-use the models created by other users.

## Documentation

The documentation is available in [FEDOT.Docs](https://itmo-nss-team.github.io/FEDOT.Docs) repository.

The description and source code of underlying algorithms is available in [FEDOT.Algs](https://github.com/ITMO-NSS-team/FEDOT.Algs) repository and its [wiki pages](https://github.com/ITMO-NSS-team/FEDOT.Algs/wiki) (in Russian).
