# FEDOT

Добавить больше баджей, отформатировать под табличку

- Версия питона, релиза в pypi
- Статус билда, coverage, docs
- Лицензия

[![Build Status](https://travis-ci.com/nccr-itmo/FEDOT.svg?token=ABTJ8bEXZokRxF3wLrtJ&branch=master)](https://travis-ci.com/nccr-itmo/FEDOT) [![Coverage Status](https://coveralls.io/repos/github/nccr-itmo/FEDOT/badge.svg?branch=master)](https://coveralls.io/github/nccr-itmo/FEDOT?branch=master)


This repository contains Fedot - a framework for automated modeling and machine learning. 
It can build composite models for the different real-world processes in an automated way using an evolutionary approach. 

Composite models - the models with heterogeneous graph-based structure, that can consist of ML models, domain-specific models, equation-based models, statistical, and even other composite models. Composite modelling allows obtaining efficient multi-scale solutions for various applied problems.

Fedot can be used for classification, regression, clustering, time series forecasting, and other similar tasks. Also, the derived solutions for other problems (e.g. bayesian generation of synthetic data) can be build using Fedot.Core.

The project is maintained by the research team of Natural Systems Simulation Lab, which is a part of the National Center for Cognitive Research of ITMO University.

## Installation

```python
git clone https://github.com/nccr-itmo/FEDOT.git
cd FEDOT
pip install -r requirements.txt
```
## FEDOT features


- The generation of high-quality variable-shaped machine learning pipelines for various tasks: binary/multiclass classification, regression, clustering, time series forecasting;
- The structural learning of composite models with different nature (hybrid, bayesian, deep learning, etc) using custom metrics;
- The seamless integration of the custom models (including domain-specific), frameworks and algorithms into pipelines;
- Benchmarking utilities that can run real-world cases  (the ready-to-use examples are provided for credit scoring, sea surface height forecasting, oil production forecasting, etc), state-of-the-art-datasets (like PMLB) and synthetic data.;

## How to use

The main purpose of FEDOT is to identify a suitable composite model for a given dataset.
The model is obtained via optimization process (we also call it 'composing').\
Firstly, you need to prepare datasets for fit and validate and specify a task
that you going to solve:
```python
task = Task(TaskTypesEnum.classification)
dataset_to_compose = InputData.from_csv(train_file_path, task=task)
dataset_to_validate = InputData.from_csv(test_file_path, task=task)
```
Then, chose a set of models that can be included in the composite model, and the optimized metric function:
```python
available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)
metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)
```
Next, you need to specify requirements for composer.
In this case, GPComposer is chosen that is based on evolutionary algorithm.
```python
composer_requirements = GPComposerRequirements(
    primary=available_model_types,
    secondary=available_model_types, max_arity=3,
    max_depth=3, pop_size=20, num_of_generations=20,
    crossover_prob=0.8, mutation_prob=0.8, max_lead_time=20)
composer = GPComposer()
```
Now you can run the optimization and obtain a composite model:
```python
chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                            initial_chain=None,
                                            composer_requirements=composer_requirements,
                                            metrics=metric_function,
                                            is_visualise=False)
```
Finally, you can test the resulted model on the validation dataset:
```python
roc_on_valid_evo_composed = calculate_validation_metric(chain_evo_composed,
                                                        dataset_to_validate)
print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
```
Extended examples:
- Credit scoring problem, i.e. binary classification task - [here](https://github.com/nccr-itmo/FEDOT/blob/master/cases/credit_scoring_problem.py)
- Time series forecasting, i.e. regression - [here](https://github.com/nccr-itmo/FEDOT/blob/master/cases/metocean_forecasting_problem.py)

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
