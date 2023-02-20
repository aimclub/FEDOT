Automated Design of Pipelines
=============================

As discussed in :doc:`architecture`, FEDOT consists, on the one hand, from modules governing machine learning and data processing, and on the other hand, from core optimization modules of GOLEM framework.

FEDOT exposes an API for controlling the optimization process. API is designed for common use-cases of ML tasks. It delegates the optimization to underlying GOLEM core with default parameters adjusted for the task.

Example of running optimization through the API can be found in the `api classification example <https://github.com/aimclub/FEDOT/blob/master/examples/simple/classification/api_classification.py>`_.

If instead users need to customize the optimization algorithm (e.g. with custom genetic operators like mutations or crossover or custom verification rules) then it's possible by directly using `ComposerBuilder` class or one of the optimizers from GOLEM.

Example of a customized usage can be found in `credit scoring case problem <https://github.com/aimclub/FEDOT/blob/master/cases/credit_scoring/credit_scoring_problem_multiobj.py>`_.
