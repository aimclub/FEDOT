# FEDOT

Добавить больше баджей, отформатировать под табличку

- Версия питона, релиза в pypi
- Статус билда, coverage, docs
- Лицензия

[![Build Status](https://travis-ci.com/nccr-itmo/FEDOT.svg?token=ABTJ8bEXZokRxF3wLrtJ&branch=master)](https://travis-ci.com/nccr-itmo/FEDOT) [![Coverage Status](https://coveralls.io/repos/github/nccr-itmo/FEDOT/badge.svg?branch=master)](https://coveralls.io/github/nccr-itmo/FEDOT?branch=master)



// Здесь надо переписать, чтобы получился мини-абстракт того, что умеет FEDOT

This repository contains the framework for the knowledge-enriched AutoML named FEDOT (Russian: Федот).
It can be used to generate high-quality composite models for the classification, regression, clustering, time series forecasting, and other real-world problems in an automated way.

The project is maintained by the research team of Natural Systems Simulation Lab, which is a part of the National Center for Cognitive Research of ITMO University.


## Installation

```python
git clone https://github.com/nccr-itmo/FEDOT.git
cd FEDOT
pip install -r requirements.txt
```
## FEDOT features


// Здесь перечислим буллетпоинты, что (а) Федот позволяет делать и (б) что он делает хорошо

To combine the proposed concepts and methods with the existing state-of-the-art approaches and share the obtained experience with the community, we decided to develop the FEDOT framework.

The framework kernel can be configured for different classes of tasks. The framework includes the library with implementations of intelligent algorithms to identify data-driven models with different requirements; and composite models (chains of models) for solving specific subject tasks (social and financial, metocean, physical, etc.).

It is possible to obtain models with given parameters of quality, complexity, interpretability; to get an any-time result; to pause and resume model identification; to integrate many popular Python open source solutions for AutoML/meta-learning, optimization, quality assessment, etc.; re-use the models created by other users.

## Examples/How To Use
// Тут будут короткие примеры кода, как запустить FEDOT

// Можно добавить ссылок на видео туториалы, еще что-то

## Project structure

// Здесь, наверное, можно коротко описать основные модули, архитектуру и сослаться на документацию


## Basic Concepts

// Здесь можно описать основные термины и концепты, которые мы включаем в FEDOT: композер, primary/secondary узлы, цепочки, 
 
// можно добавить ссылки на всякие наши статьи
 

## Current R&D and future plans

// Здесь можно коротко описать, над чем мы сейчас работаем 

## Documentation

The documentation is available in [FEDOT.Docs](https://itmo-nss-team.github.io/FEDOT.Docs) repository.

The description and source code of underlying algorithms is available in [FEDOT.Algs](https://github.com/ITMO-NSS-team/FEDOT.Algs) repository and its [wiki pages](https://github.com/ITMO-NSS-team/FEDOT.Algs/wiki) (in Russian).

// Дополнительно будет ссылка на readthedocs

## Contribution Guide

## Acknowledgements

### Supported by
- [National Center for Cognitive Research of ITMO University](https://actcognitive.org/)

### Citation

latex-ссылка на основную статью про фреймворк (когда она появится)