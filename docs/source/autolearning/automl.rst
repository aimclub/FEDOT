Modern AutoML Solutions
=======================

**State-of-the-art AutoML solutions: a brief overview**

We cannot deny that there are a lot of well-implemented AutoML
frameworks that are quite popular.

There is a brief list of examples of existing solutions:
* Autosklearn
* AutoKeras
* H2O
* Google Colud HyperTune
* Microsoft Automated ML
* TPOP
* Hyperopt
* PyBrain

We tried to summarize the main advantages and disadvantages of this
solutions to the breif table:

.. figure::  img/autoMLsolutions.png
   :align:   center

However, the modern AutoML is mostly focused on relatively simple tasks
of hyperparameters optimization, input data preprocessing, selecting a
single model or a set of models [1] (this approach is also referred to
as the Combined Algorithm Selection and Hyperparameters optimization -
CASH) since the overall learning and meta-learning process is extremely
expensive.

Sources
-------

[1] Xin He, Kaiyong Zhao, and Xiaowen Chu. 2019. AutoML: A Survey of the
State-of-the-Art.arXiv preprint arXiv:1908.00709(2019).
